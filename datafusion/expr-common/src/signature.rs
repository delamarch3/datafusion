// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Signature module contains foundational types that are used to represent signatures, types,
//! and return types of functions in DataFusion.

use crate::type_coercion::{
    aggregates::{ALL, NUMERICS, STRINGS},
    binary::{binary_numeric_coercion, comparison_coercion, string_coercion},
};
use arrow::{compute::can_cast_types, datatypes::DataType};
use datafusion_common::{
    internal_datafusion_err, internal_err, plan_err,
    utils::coerced_fixed_size_list_to_list, Result,
};
use itertools::Itertools;

/// Constant that is used as a placeholder for any valid timezone.
/// This is used where a function can accept a timestamp type with any
/// valid timezone, it exists to avoid the need to enumerate all possible
/// timezones. See [`TypeSignature`] for more details.
///
/// Type coercion always ensures that functions will be executed using
/// timestamp arrays that have a valid time zone. Functions must never
/// return results with this timezone.
pub const TIMEZONE_WILDCARD: &str = "+TZ";

/// Constant that is used as a placeholder for any valid fixed size list.
/// This is used where a function can accept a fixed size list type with any
/// valid length. It exists to avoid the need to enumerate all possible fixed size list lengths.
pub const FIXED_SIZE_LIST_WILDCARD: i32 = i32::MIN;

/// A function's volatility, which defines the functions eligibility for certain optimizations
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum Volatility {
    /// An immutable function will always return the same output when given the same
    /// input. DataFusion will attempt to inline immutable functions during planning.
    Immutable,
    /// A stable function may return different values given the same input across different
    /// queries but must return the same value for a given input within a query. An example of
    /// this is the `Now` function. DataFusion will attempt to inline `Stable` functions
    /// during planning, when possible.
    /// For query `select col1, now() from t1`, it might take a while to execute but
    /// `now()` column will be the same for each output row, which is evaluated
    /// during planning.
    Stable,
    /// A volatile function may change the return value from evaluation to evaluation.
    /// Multiple invocations of a volatile function may return different results when used in the
    /// same query. An example of this is the random() function. DataFusion
    /// can not evaluate such functions during planning.
    /// In the query `select col1, random() from t1`, `random()` function will be evaluated
    /// for each output row, resulting in a unique random value for each row.
    Volatile,
}

/// A function's type signature defines the types of arguments the function supports.
///
/// Functions typically support only a few different types of arguments compared to the
/// different datatypes in Arrow. To make functions easy to use, when possible DataFusion
/// automatically coerces (add casts to) function arguments so they match the type signature.
///
/// For example, a function like `cos` may only be implemented for `Float64` arguments. To support a query
/// that calls `cos` with a different argument type, such as `cos(int_column)`, type coercion automatically
/// adds a cast such as `cos(CAST int_column AS DOUBLE)` during planning.
///
/// # Data Types
/// Types to match are represented using Arrow's [`DataType`].  [`DataType::Timestamp`] has an optional variable
/// timezone specification. To specify a function can handle a timestamp with *ANY* timezone, use
/// the [`TIMEZONE_WILDCARD`]. For example:
///
/// ```
/// # use arrow::datatypes::{DataType, TimeUnit};
/// # use datafusion_expr_common::signature::{TIMEZONE_WILDCARD, TypeSignature};
/// let type_signature = TypeSignature::Exact(vec![
///   // A nanosecond precision timestamp with ANY timezone
///   // matches  Timestamp(Nanosecond, Some("+0:00"))
///   // matches  Timestamp(Nanosecond, Some("+5:00"))
///   // does not match  Timestamp(Nanosecond, None)
///   DataType::Timestamp(TimeUnit::Nanosecond, Some(TIMEZONE_WILDCARD.into())),
/// ]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum TypeSignature {
    /// One or more arguments of a common type out of a list of valid types.
    ///
    /// # Examples
    /// A function such as `concat` is `Variadic(vec![DataType::Utf8, DataType::LargeUtf8])`
    Variadic(Vec<DataType>),
    /// The acceptable signature and coercions rules to coerce arguments to this
    /// signature are special for this function. If this signature is specified,
    /// DataFusion will call `ScalarUDFImpl::coerce_types` to prepare argument types.
    UserDefined,
    /// One or more arguments with arbitrary types
    VariadicAny,
    /// Fixed number of arguments of an arbitrary but equal type out of a list of valid types.
    ///
    /// # Examples
    /// 1. A function of one argument of f64 is `Uniform(1, vec![DataType::Float64])`
    /// 2. A function of one argument of f64 or f32 is `Uniform(1, vec![DataType::Float32, DataType::Float64])`
    Uniform(usize, Vec<DataType>),
    /// Exact number of arguments of an exact type
    Exact(Vec<DataType>),
    /// The number of arguments that can be coerced to in order
    /// For example, `Coercible(vec![DataType::Float64])` accepts
    /// arguments like `vec![DataType::Int32]` or `vec![DataType::Float32]`
    /// since i32 and f32 can be casted to f64
    Coercible(Vec<DataType>),
    /// Fixed number of arguments of arbitrary types
    /// If a function takes 0 argument, its `TypeSignature` should be `Any(0)`
    Any(usize),
    /// Matches exactly one of a list of [`TypeSignature`]s. Coercion is attempted to match
    /// the signatures in order, and stops after the first success, if any.
    ///
    /// # Examples
    /// Function `make_array` takes 0 or more arguments with arbitrary types, its `TypeSignature`
    /// is `OneOf(vec![Any(0), VariadicAny])`.
    OneOf(Vec<TypeSignature>),
    /// Specifies Signatures for array functions
    ArraySignature(ArrayFunctionSignature),
    /// Fixed number of arguments of numeric types.
    /// See <https://docs.rs/arrow/latest/arrow/datatypes/enum.DataType.html#method.is_numeric> to know which type is considered numeric
    Numeric(usize),
    /// Fixed number of arguments of all the same string types.
    /// The precedence of type from high to low is Utf8View, LargeUtf8 and Utf8.
    /// Null is considerd as `Utf8` by default
    /// Dictionary with string value type is also handled.
    String(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum ArrayFunctionSignature {
    /// Specialized Signature for ArrayAppend and similar functions
    /// The first argument should be List/LargeList/FixedSizedList, and the second argument should be non-list or list.
    /// The second argument's list dimension should be one dimension less than the first argument's list dimension.
    /// List dimension of the List/LargeList is equivalent to the number of List.
    /// List dimension of the non-list is 0.
    ArrayAndElement,
    /// Specialized Signature for ArrayPrepend and similar functions
    /// The first argument should be non-list or list, and the second argument should be List/LargeList.
    /// The first argument's list dimension should be one dimension less than the second argument's list dimension.
    ElementAndArray,
    /// Specialized Signature for Array functions of the form (List/LargeList, Index)
    /// The first argument should be List/LargeList/FixedSizedList, and the second argument should be Int64.
    ArrayAndIndex,
    /// Specialized Signature for Array functions of the form (List/LargeList, Element, Optional Index)
    ArrayAndElementAndOptionalIndex,
    /// Specialized Signature for ArrayEmpty and similar functions
    /// The function takes a single argument that must be a List/LargeList/FixedSizeList
    /// or something that can be coerced to one of those types.
    Array,
    /// Specialized Signature for MapArray
    /// The function takes a single argument that must be a MapArray
    MapArray,
}

impl std::fmt::Display for ArrayFunctionSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrayFunctionSignature::ArrayAndElement => {
                write!(f, "array, element")
            }
            ArrayFunctionSignature::ArrayAndElementAndOptionalIndex => {
                write!(f, "array, element, [index]")
            }
            ArrayFunctionSignature::ElementAndArray => {
                write!(f, "element, array")
            }
            ArrayFunctionSignature::ArrayAndIndex => {
                write!(f, "array, index")
            }
            ArrayFunctionSignature::Array => {
                write!(f, "array")
            }
            ArrayFunctionSignature::MapArray => {
                write!(f, "map_array")
            }
        }
    }
}

impl TypeSignature {
    pub fn to_string_repr(&self) -> Vec<String> {
        match self {
            TypeSignature::Variadic(types) => {
                vec![format!("{}, ..", Self::join_types(types, "/"))]
            }
            TypeSignature::Uniform(arg_count, valid_types) => {
                vec![std::iter::repeat(Self::join_types(valid_types, "/"))
                    .take(*arg_count)
                    .collect::<Vec<String>>()
                    .join(", ")]
            }
            TypeSignature::String(num) => {
                vec![format!("String({num})")]
            }
            TypeSignature::Numeric(num) => {
                vec![format!("Numeric({num})")]
            }
            TypeSignature::Exact(types) | TypeSignature::Coercible(types) => {
                vec![Self::join_types(types, ", ")]
            }
            TypeSignature::Any(arg_count) => {
                vec![std::iter::repeat("Any")
                    .take(*arg_count)
                    .collect::<Vec<&str>>()
                    .join(", ")]
            }
            TypeSignature::UserDefined => {
                vec!["UserDefined".to_string()]
            }
            TypeSignature::VariadicAny => vec!["Any, .., Any".to_string()],
            TypeSignature::OneOf(sigs) => {
                sigs.iter().flat_map(|s| s.to_string_repr()).collect()
            }
            TypeSignature::ArraySignature(array_signature) => {
                vec![array_signature.to_string()]
            }
        }
    }

    /// Helper function to join types with specified delimiter.
    pub fn join_types<T: std::fmt::Display>(types: &[T], delimiter: &str) -> String {
        types
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<String>>()
            .join(delimiter)
    }

    /// Check whether 0 input argument is valid for given `TypeSignature`
    pub fn supports_zero_argument(&self) -> bool {
        match &self {
            TypeSignature::Exact(vec) => vec.is_empty(),
            TypeSignature::Uniform(0, _) | TypeSignature::Any(0) => true,
            TypeSignature::OneOf(types) => types
                .iter()
                .any(|type_sig| type_sig.supports_zero_argument()),
            _ => false,
        }
    }

    /// get all possible types for the given `TypeSignature`
    pub fn get_possible_types(&self) -> Vec<Vec<DataType>> {
        match self {
            TypeSignature::Exact(types) => vec![types.clone()],
            TypeSignature::OneOf(types) => types
                .iter()
                .flat_map(|type_sig| type_sig.get_possible_types())
                .collect(),
            TypeSignature::Uniform(arg_count, types) => types
                .iter()
                .cloned()
                .map(|data_type| vec![data_type; *arg_count])
                .collect(),
            TypeSignature::Coercible(types) => types
                .iter()
                .map(|to_type| {
                    ALL.iter()
                        .filter(|from_type| can_cast_types(from_type, to_type))
                        .cloned()
                        .collect::<Vec<DataType>>()
                })
                .multi_cartesian_product()
                .collect(),
            TypeSignature::Variadic(types) => types
                .iter()
                .cloned()
                .map(|data_type| vec![data_type; types.len()])
                .collect(),
            TypeSignature::Numeric(arg_count) => NUMERICS
                .iter()
                .cloned()
                .map(|numeric_type| vec![numeric_type; *arg_count])
                .collect(),
            TypeSignature::String(arg_count) => STRINGS
                .iter()
                .cloned()
                .map(|string_type| vec![string_type; *arg_count])
                .collect(),
            // TODO: Implement for other types
            TypeSignature::Any(_)
            | TypeSignature::VariadicAny
            | TypeSignature::ArraySignature(_)
            | TypeSignature::UserDefined => vec![],
        }
    }
}

/// Defines the supported argument types ([`TypeSignature`]) and [`Volatility`] for a function.
///
/// DataFusion will automatically coerce (cast) argument types to one of the supported
/// function signatures, if possible.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct Signature {
    /// The data types that the function accepts. See [TypeSignature] for more information.
    pub type_signature: TypeSignature,
    /// The volatility of the function. See [Volatility] for more information.
    pub volatility: Volatility,
}

impl Signature {
    /// Creates a new Signature from a given type signature and volatility.
    pub fn new(type_signature: TypeSignature, volatility: Volatility) -> Self {
        Signature {
            type_signature,
            volatility,
        }
    }
    /// An arbitrary number of arguments with the same type, from those listed in `common_types`.
    pub fn variadic(common_types: Vec<DataType>, volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::Variadic(common_types),
            volatility,
        }
    }
    /// User-defined coercion rules for the function.
    pub fn user_defined(volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::UserDefined,
            volatility,
        }
    }

    /// A specified number of numeric arguments
    pub fn numeric(arg_count: usize, volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::Numeric(arg_count),
            volatility,
        }
    }

    /// A specified number of numeric arguments
    pub fn string(arg_count: usize, volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::String(arg_count),
            volatility,
        }
    }

    /// An arbitrary number of arguments of any type.
    pub fn variadic_any(volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::VariadicAny,
            volatility,
        }
    }
    /// A fixed number of arguments of the same type, from those listed in `valid_types`.
    pub fn uniform(
        arg_count: usize,
        valid_types: Vec<DataType>,
        volatility: Volatility,
    ) -> Self {
        Self {
            type_signature: TypeSignature::Uniform(arg_count, valid_types),
            volatility,
        }
    }
    /// Exactly matches the types in `exact_types`, in order.
    pub fn exact(exact_types: Vec<DataType>, volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::Exact(exact_types),
            volatility,
        }
    }
    /// Target coerce types in order
    pub fn coercible(target_types: Vec<DataType>, volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::Coercible(target_types),
            volatility,
        }
    }

    /// A specified number of arguments of any type
    pub fn any(arg_count: usize, volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::Any(arg_count),
            volatility,
        }
    }
    /// Any one of a list of [TypeSignature]s.
    pub fn one_of(type_signatures: Vec<TypeSignature>, volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::OneOf(type_signatures),
            volatility,
        }
    }
    /// Specialized Signature for ArrayAppend and similar functions
    pub fn array_and_element(volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::ArraySignature(
                ArrayFunctionSignature::ArrayAndElement,
            ),
            volatility,
        }
    }
    /// Specialized Signature for Array functions with an optional index
    pub fn array_and_element_and_optional_index(volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::ArraySignature(
                ArrayFunctionSignature::ArrayAndElementAndOptionalIndex,
            ),
            volatility,
        }
    }
    /// Specialized Signature for ArrayPrepend and similar functions
    pub fn element_and_array(volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::ArraySignature(
                ArrayFunctionSignature::ElementAndArray,
            ),
            volatility,
        }
    }
    /// Specialized Signature for ArrayElement and similar functions
    pub fn array_and_index(volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::ArraySignature(
                ArrayFunctionSignature::ArrayAndIndex,
            ),
            volatility,
        }
    }
    /// Specialized Signature for ArrayEmpty and similar functions
    pub fn array(volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::ArraySignature(ArrayFunctionSignature::Array),
            volatility,
        }
    }
}

/// Returns a Vec of all possible valid argument types for the given signature.
pub fn get_valid_types(
    signature: &TypeSignature,
    current_types: &[DataType],
) -> Result<Vec<Vec<DataType>>> {
    fn array_element_and_optional_index(
        current_types: &[DataType],
    ) -> Result<Vec<Vec<DataType>>> {
        // make sure there's 2 or 3 arguments
        if !(current_types.len() == 2 || current_types.len() == 3) {
            return Ok(vec![vec![]]);
        }

        let first_two_types = &current_types[0..2];
        let mut valid_types = array_append_or_prepend_valid_types(first_two_types, true)?;

        // Early return if there are only 2 arguments
        if current_types.len() == 2 {
            return Ok(valid_types);
        }

        let valid_types_with_index = valid_types
            .iter()
            .map(|t| {
                let mut t = t.clone();
                t.push(DataType::Int64);
                t
            })
            .collect::<Vec<_>>();

        valid_types.extend(valid_types_with_index);

        Ok(valid_types)
    }

    fn array_append_or_prepend_valid_types(
        current_types: &[DataType],
        is_append: bool,
    ) -> Result<Vec<Vec<DataType>>> {
        if current_types.len() != 2 {
            return Ok(vec![vec![]]);
        }

        let (array_type, elem_type) = if is_append {
            (&current_types[0], &current_types[1])
        } else {
            (&current_types[1], &current_types[0])
        };

        // We follow Postgres on `array_append(Null, T)`, which is not valid.
        if array_type.eq(&DataType::Null) {
            return Ok(vec![vec![]]);
        }

        // We need to find the coerced base type, mainly for cases like:
        // `array_append(List(null), i64)` -> `List(i64)`
        let array_base_type = datafusion_common::utils::base_type(array_type);
        let elem_base_type = datafusion_common::utils::base_type(elem_type);
        let new_base_type = comparison_coercion(&array_base_type, &elem_base_type);

        let new_base_type = new_base_type.ok_or_else(|| {
            internal_datafusion_err!(
                "Coercion from {array_base_type:?} to {elem_base_type:?} not supported."
            )
        })?;

        let new_array_type = datafusion_common::utils::coerced_type_with_base_type_only(
            array_type,
            &new_base_type,
        );

        match new_array_type {
            DataType::List(ref field)
            | DataType::LargeList(ref field)
            | DataType::FixedSizeList(ref field, _) => {
                let new_elem_type = field.data_type();
                if is_append {
                    Ok(vec![vec![new_array_type.clone(), new_elem_type.clone()]])
                } else {
                    Ok(vec![vec![new_elem_type.to_owned(), new_array_type.clone()]])
                }
            }
            _ => Ok(vec![vec![]]),
        }
    }
    fn array(array_type: &DataType) -> Option<DataType> {
        match array_type {
            DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(_, _) => {
                let array_type = coerced_fixed_size_list_to_list(array_type);
                Some(array_type)
            }
            _ => None,
        }
    }

    let valid_types = match signature {
        TypeSignature::Variadic(valid_types) => valid_types
            .iter()
            .map(|valid_type| current_types.iter().map(|_| valid_type.clone()).collect())
            .collect(),
        TypeSignature::String(number) => {
            if *number < 1 {
                return plan_err!(
                    "The signature expected at least one argument but received {}",
                    current_types.len()
                );
            }
            if *number != current_types.len() {
                return plan_err!(
                    "The signature expected {} arguments but received {}",
                    number,
                    current_types.len()
                );
            }

            fn coercion_rule(
                lhs_type: &DataType,
                rhs_type: &DataType,
            ) -> Result<DataType> {
                match (lhs_type, rhs_type) {
                    (DataType::Null, DataType::Null) => Ok(DataType::Utf8),
                    (DataType::Null, data_type) | (data_type, DataType::Null) => {
                        coercion_rule(data_type, &DataType::Utf8)
                    }
                    (DataType::Dictionary(_, lhs), DataType::Dictionary(_, rhs)) => {
                        coercion_rule(lhs, rhs)
                    }
                    (DataType::Dictionary(_, v), other)
                    | (other, DataType::Dictionary(_, v)) => coercion_rule(v, other),
                    _ => {
                        if let Some(coerced_type) = string_coercion(lhs_type, rhs_type) {
                            Ok(coerced_type)
                        } else {
                            plan_err!(
                                "{} and {} are not coercible to a common string type",
                                lhs_type,
                                rhs_type
                            )
                        }
                    }
                }
            }

            // Length checked above, safe to unwrap
            let mut coerced_type = current_types.first().unwrap().to_owned();
            for t in current_types.iter().skip(1) {
                coerced_type = coercion_rule(&coerced_type, t)?;
            }

            fn base_type_or_default_type(data_type: &DataType) -> DataType {
                if data_type.is_null() {
                    DataType::Utf8
                } else if let DataType::Dictionary(_, v) = data_type {
                    base_type_or_default_type(v)
                } else {
                    data_type.to_owned()
                }
            }

            vec![vec![base_type_or_default_type(&coerced_type); *number]]
        }
        TypeSignature::Numeric(number) => {
            if *number < 1 {
                return plan_err!(
                    "The signature expected at least one argument but received {}",
                    current_types.len()
                );
            }
            if *number != current_types.len() {
                return plan_err!(
                    "The signature expected {} arguments but received {}",
                    number,
                    current_types.len()
                );
            }

            let mut valid_type = current_types.first().unwrap().clone();
            for t in current_types.iter().skip(1) {
                if let Some(coerced_type) = binary_numeric_coercion(&valid_type, t) {
                    valid_type = coerced_type;
                } else {
                    return plan_err!(
                        "{} and {} are not coercible to a common numeric type",
                        valid_type,
                        t
                    );
                }
            }

            vec![vec![valid_type; *number]]
        }
        TypeSignature::Coercible(target_types) => {
            if target_types.is_empty() {
                return plan_err!(
                    "The signature expected at least one argument but received {}",
                    current_types.len()
                );
            }
            if target_types.len() != current_types.len() {
                return plan_err!(
                    "The signature expected {} arguments but received {}",
                    target_types.len(),
                    current_types.len()
                );
            }

            for (data_type, target_type) in current_types.iter().zip(target_types.iter())
            {
                if !can_cast_types(data_type, target_type) {
                    return plan_err!("{data_type} is not coercible to {target_type}");
                }
            }

            vec![target_types.to_owned()]
        }
        TypeSignature::Uniform(number, valid_types) => valid_types
            .iter()
            .map(|valid_type| (0..*number).map(|_| valid_type.clone()).collect())
            .collect(),
        TypeSignature::UserDefined => {
            return internal_err!(
            "User-defined signature should be handled by function-specific coerce_types."
        )
        }
        TypeSignature::VariadicAny => {
            vec![current_types.to_vec()]
        }
        TypeSignature::Exact(valid_types) => vec![valid_types.clone()],
        TypeSignature::ArraySignature(ref function_signature) => match function_signature
        {
            ArrayFunctionSignature::ArrayAndElement => {
                array_append_or_prepend_valid_types(current_types, true)?
            }
            ArrayFunctionSignature::ElementAndArray => {
                array_append_or_prepend_valid_types(current_types, false)?
            }
            ArrayFunctionSignature::ArrayAndIndex => {
                if current_types.len() != 2 {
                    return Ok(vec![vec![]]);
                }
                array(&current_types[0]).map_or_else(
                    || vec![vec![]],
                    |array_type| vec![vec![array_type, DataType::Int64]],
                )
            }
            ArrayFunctionSignature::ArrayAndElementAndOptionalIndex => {
                array_element_and_optional_index(current_types)?
            }
            ArrayFunctionSignature::Array => {
                if current_types.len() != 1 {
                    return Ok(vec![vec![]]);
                }

                array(&current_types[0])
                    .map_or_else(|| vec![vec![]], |array_type| vec![vec![array_type]])
            }
            ArrayFunctionSignature::MapArray => {
                if current_types.len() != 1 {
                    return Ok(vec![vec![]]);
                }

                match &current_types[0] {
                    DataType::Map(_, _) => vec![vec![current_types[0].clone()]],
                    _ => vec![vec![]],
                }
            }
        },
        TypeSignature::Any(number) => {
            if current_types.len() != *number {
                return plan_err!(
                    "The function expected {} arguments but received {}",
                    number,
                    current_types.len()
                );
            }
            vec![(0..*number).map(|i| current_types[i].clone()).collect()]
        }
        TypeSignature::OneOf(types) => types
            .iter()
            .filter_map(|t| get_valid_types(t, current_types).ok())
            .flatten()
            .collect::<Vec<_>>(),
    };

    Ok(valid_types)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_zero_argument_tests() {
        // Testing `TypeSignature`s which supports 0 arg
        let positive_cases = vec![
            TypeSignature::Exact(vec![]),
            TypeSignature::Uniform(0, vec![DataType::Float64]),
            TypeSignature::Any(0),
            TypeSignature::OneOf(vec![
                TypeSignature::Exact(vec![DataType::Int8]),
                TypeSignature::Any(0),
                TypeSignature::Uniform(1, vec![DataType::Int8]),
            ]),
        ];

        for case in positive_cases {
            assert!(
                case.supports_zero_argument(),
                "Expected {:?} to support zero arguments",
                case
            );
        }

        // Testing `TypeSignature`s which doesn't support 0 arg
        let negative_cases = vec![
            TypeSignature::Exact(vec![DataType::Utf8]),
            TypeSignature::Uniform(1, vec![DataType::Float64]),
            TypeSignature::Any(1),
            TypeSignature::VariadicAny,
            TypeSignature::OneOf(vec![
                TypeSignature::Exact(vec![DataType::Int8]),
                TypeSignature::Uniform(1, vec![DataType::Int8]),
            ]),
        ];

        for case in negative_cases {
            assert!(
                !case.supports_zero_argument(),
                "Expected {:?} not to support zero arguments",
                case
            );
        }
    }

    #[test]
    fn type_signature_partial_ord() {
        // Test validates that partial ord is defined for TypeSignature and Signature.
        assert!(TypeSignature::UserDefined < TypeSignature::VariadicAny);
        assert!(TypeSignature::UserDefined < TypeSignature::Any(1));

        assert!(
            TypeSignature::Uniform(1, vec![DataType::Null])
                < TypeSignature::Uniform(1, vec![DataType::Boolean])
        );
        assert!(
            TypeSignature::Uniform(1, vec![DataType::Null])
                < TypeSignature::Uniform(2, vec![DataType::Null])
        );
        assert!(
            TypeSignature::Uniform(usize::MAX, vec![DataType::Null])
                < TypeSignature::Exact(vec![DataType::Null])
        );
    }

    #[test]
    fn test_get_possible_types() {
        let type_signature = TypeSignature::Exact(vec![DataType::Int32, DataType::Int64]);
        let possible_types = type_signature.get_possible_types();
        assert_eq!(possible_types, vec![vec![DataType::Int32, DataType::Int64]]);

        let type_signature = TypeSignature::OneOf(vec![
            TypeSignature::Exact(vec![DataType::Int32, DataType::Int64]),
            TypeSignature::Exact(vec![DataType::Float32, DataType::Float64]),
        ]);
        let possible_types = type_signature.get_possible_types();
        assert_eq!(
            possible_types,
            vec![
                vec![DataType::Int32, DataType::Int64],
                vec![DataType::Float32, DataType::Float64]
            ]
        );

        let type_signature = TypeSignature::OneOf(vec![
            TypeSignature::Exact(vec![DataType::Int32, DataType::Int64]),
            TypeSignature::Exact(vec![DataType::Float32, DataType::Float64]),
            TypeSignature::Exact(vec![DataType::Utf8]),
        ]);
        let possible_types = type_signature.get_possible_types();
        assert_eq!(
            possible_types,
            vec![
                vec![DataType::Int32, DataType::Int64],
                vec![DataType::Float32, DataType::Float64],
                vec![DataType::Utf8]
            ]
        );

        let type_signature =
            TypeSignature::Uniform(2, vec![DataType::Float32, DataType::Int64]);
        let possible_types = type_signature.get_possible_types();
        assert_eq!(
            possible_types,
            vec![
                vec![DataType::Float32, DataType::Float32],
                vec![DataType::Int64, DataType::Int64]
            ]
        );

        let type_signature =
            TypeSignature::Coercible(vec![DataType::Utf8, DataType::Int64]);
        let possible_types = type_signature.get_possible_types();
        assert_eq!(possible_types.len(), 255);
        let possible_utf8_first_types = possible_types
            .iter()
            .filter(|types| types[0] == DataType::Utf8)
            .cloned()
            .collect::<Vec<Vec<DataType>>>();
        assert_eq!(
            possible_utf8_first_types,
            vec![
                vec![DataType::Utf8, DataType::Utf8],
                vec![DataType::Utf8, DataType::LargeUtf8],
                vec![DataType::Utf8, DataType::Int8],
                vec![DataType::Utf8, DataType::Int16],
                vec![DataType::Utf8, DataType::Int32],
                vec![DataType::Utf8, DataType::Int64],
                vec![DataType::Utf8, DataType::UInt8],
                vec![DataType::Utf8, DataType::UInt16],
                vec![DataType::Utf8, DataType::UInt32],
                vec![DataType::Utf8, DataType::UInt64],
                vec![DataType::Utf8, DataType::Float16],
                vec![DataType::Utf8, DataType::Float32],
                vec![DataType::Utf8, DataType::Float64],
                vec![DataType::Utf8, DataType::Date32],
                vec![DataType::Utf8, DataType::Date64]
            ]
        );

        let type_signature =
            TypeSignature::Variadic(vec![DataType::Int32, DataType::Int64]);
        let possible_types = type_signature.get_possible_types();
        assert_eq!(
            possible_types,
            vec![
                vec![DataType::Int32, DataType::Int32],
                vec![DataType::Int64, DataType::Int64]
            ]
        );

        let type_signature = TypeSignature::Numeric(2);
        let possible_types = type_signature.get_possible_types();
        assert_eq!(
            possible_types,
            vec![
                vec![DataType::Int8, DataType::Int8],
                vec![DataType::Int16, DataType::Int16],
                vec![DataType::Int32, DataType::Int32],
                vec![DataType::Int64, DataType::Int64],
                vec![DataType::UInt8, DataType::UInt8],
                vec![DataType::UInt16, DataType::UInt16],
                vec![DataType::UInt32, DataType::UInt32],
                vec![DataType::UInt64, DataType::UInt64],
                vec![DataType::Float32, DataType::Float32],
                vec![DataType::Float64, DataType::Float64]
            ]
        );

        let type_signature = TypeSignature::String(2);
        let possible_types = type_signature.get_possible_types();
        assert_eq!(
            possible_types,
            vec![
                vec![DataType::Utf8, DataType::Utf8],
                vec![DataType::LargeUtf8, DataType::LargeUtf8]
            ]
        );
    }

    #[test]
    fn test_get_valid_types_one_of() -> Result<()> {
        let signature =
            TypeSignature::OneOf(vec![TypeSignature::Any(1), TypeSignature::Any(2)]);

        let invalid_types = get_valid_types(
            &signature,
            &[DataType::Int32, DataType::Int32, DataType::Int32],
        )?;
        assert_eq!(invalid_types.len(), 0);

        let args = vec![DataType::Int32, DataType::Int32];
        let valid_types = get_valid_types(&signature, &args)?;
        assert_eq!(valid_types.len(), 1);
        assert_eq!(valid_types[0], args);

        let args = vec![DataType::Int32];
        let valid_types = get_valid_types(&signature, &args)?;
        assert_eq!(valid_types.len(), 1);
        assert_eq!(valid_types[0], args);

        Ok(())
    }
}
