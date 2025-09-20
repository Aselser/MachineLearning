## Naïve Bayes Classifier
## 📘 Theory: Naïve Bayes Classifier

The **Naïve Bayes classifier** is a probabilistic model based on **Bayes’ Theorem**, with the “naïve” assumption that all features are conditionally independent given the class.

### Bayes’ Theorem
For a class C and features (x₁, x₂, …, xₙ):

P(C | x₁, x₂, …, xₙ) = [ P(C) × Π P(xᵢ | C) ] / P(x₁, x₂, …, xₙ)

- **P(C):** prior probability of class C  
- **P(xᵢ | C):** likelihood of feature xᵢ given class C  
- **P(C | x₁, …, xₙ):** posterior probability of class C given the features  

---

### Classification Rule
We assign the class with the highest posterior probability:

Ĉ = argmax₍C₎ [ P(C) × Π P(xᵢ | C) ]

Since the denominator P(x₁, …, xₙ) is constant for all classes, it can be ignored in practice.

---

### Laplace Smoothing
If a feature value never appears in training data for a given class, its probability would be zero.  
To avoid this, we use **Laplace smoothing**:

P(xᵢ | C) = ( count(xᵢ, C) + 1 ) / ( count(C) + k )

- **count(xᵢ, C):** how many times feature value xᵢ appears within class C  
- **count(C):** number of samples in class C  
- **k:** number of possible values for the feature  
