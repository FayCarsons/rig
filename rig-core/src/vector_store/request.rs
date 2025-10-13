use serde::{Deserialize, Serialize};

use super::VectorStoreError;

/// A vector search request - used in the [`super::VectorStoreIndex`] trait.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VectorSearchRequest<Filter = ConcreteFilter> {
    /// The query to be embedded and used in similarity search.
    query: String,
    /// The maximum number of samples that may be returned. If adding a similarity search threshold, you may receive less than the inputted number if there aren't enough results that satisfy the threshold.
    samples: u64,
    /// Similarity search threshold. If present, any result with a distance less than this may be omitted from the final result.
    threshold: Option<f64>,
    /// Any additional parameters that are required by the vector store.
    additional_params: Option<serde_json::Value>,
    /// An expression used to filter samples
    filter: Option<Filter>,
}

impl<Filter> VectorSearchRequest<Filter>
where
    Filter: SearchFilter,
{
    /// Creates a [`VectorSearchRequestBuilder`] which you can use to instantiate this struct.
    pub fn builder() -> VectorSearchRequestBuilder<Filter> {
        VectorSearchRequestBuilder::<Filter>::default()
    }

    /// The query to be embedded and used in similarity search.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// The maximum number of samples that may be returned. If adding a similarity search threshold, you may receive less than the inputted number if there aren't enough results that satisfy the threshold.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    pub fn threshold(&self) -> Option<f64> {
        self.threshold
    }
}

pub trait SearchFilter {
    type Key;
    type Value;

    fn eq(key: Self::Key, value: Self::Value) -> Self;
    fn gt(key: Self::Key, value: Self::Value) -> Self;
    fn lt(key: Self::Key, value: Self::Value) -> Self;
    fn and(self, rhs: Self) -> Self;
    fn or(self, rhs: Self) -> Self;
}

/// A canonical, serializable retpresentation of filter expressions.
/// This serves as an intermediary form whenever you need to inspect,
/// store, or translate between specific vector store backends
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConcreteFilter {
    Eq(String, serde_json::Value),
    Gt(String, serde_json::Value),
    Lt(String, serde_json::Value),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

impl SearchFilter for ConcreteFilter {
    type Key = String;
    type Value = serde_json::Value;

    fn eq(key: Self::Key, value: Self::Value) -> Self {
        Self::Eq(key, value)
    }

    fn gt(key: Self::Key, value: Self::Value) -> Self {
        Self::Gt(key, value)
    }

    fn lt(key: Self::Key, value: Self::Value) -> Self {
        Self::Lt(key, value)
    }

    fn and(self, rhs: Self) -> Self {
        Self::And(self.into(), rhs.into())
    }

    fn or(self, rhs: Self) -> Self {
        Self::Or(self.into(), rhs.into())
    }
}

impl ConcreteFilter {
    fn interpret<F>(self) -> F
    where
        F: SearchFilter<Value = serde_json::Value>,
        F::Key: From<String>,
    {
        match self {
            Self::Eq(key, val) => F::eq(key.into(), val),
            Self::Gt(key, val) => F::gt(key.into(), val),
            Self::Lt(key, val) => F::lt(key.into(), val),
            Self::And(lhs, rhs) => F::and(lhs.interpret(), rhs.interpret()),
            Self::Or(lhs, rhs) => F::or(lhs.interpret(), rhs.interpret()),
        }
    }
}

/// The builder struct to instantiate [`VectorSearchRequest`].
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VectorSearchRequestBuilder<F = ConcreteFilter> {
    query: Option<String>,
    samples: Option<u64>,
    threshold: Option<f64>,
    additional_params: Option<serde_json::Value>,
    filter: Option<F>,
}

impl<F> Default for VectorSearchRequestBuilder<F> {
    fn default() -> Self {
        Self {
            query: None,
            samples: None,
            threshold: None,
            additional_params: None,
            filter: None,
        }
    }
}

impl<F> VectorSearchRequestBuilder<F>
where
    F: SearchFilter,
{
    /// Set the query (that will then be embedded )
    pub fn query<T>(mut self, query: T) -> Self
    where
        T: Into<String>,
    {
        self.query = Some(query.into());
        self
    }

    pub fn samples(mut self, samples: u64) -> Self {
        self.samples = Some(samples);
        self
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn additional_params(
        mut self,
        params: serde_json::Value,
    ) -> Result<Self, VectorStoreError> {
        self.additional_params = Some(params);
        Ok(self)
    }

    pub fn filter(mut self, filter: F) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn build(self) -> Result<VectorSearchRequest, VectorStoreError> {
        let Some(query) = self.query else {
            return Err(VectorStoreError::BuilderError(
                "`query` is a required variable for building a vector search request".into(),
            ));
        };

        let Some(samples) = self.samples else {
            return Err(VectorStoreError::BuilderError(
                "`samples` is a required variable for building a vector search request".into(),
            ));
        };

        let additional_params = if let Some(params) = self.additional_params {
            if !params.is_object() {
                return Err(VectorStoreError::BuilderError(
                    "Expected JSON object for additional params, got something else".into(),
                ));
            }
            Some(params)
        } else {
            None
        };

        Ok(VectorSearchRequest {
            query,
            samples,
            threshold: self.threshold,
            additional_params,
            filter: None,
        })
    }
}
