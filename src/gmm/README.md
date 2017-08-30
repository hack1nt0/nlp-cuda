# Prob : The determinant of Covariance Matrix is negative (far from zero)...

### Why? numeric underflow...

### Solution...

# Prob : likelihood is small and look the same with different K(# of clusters)

### Why? conv is not set-to-zero before accumulate Add...

### Solution...

# Prob : likelihood of some documents are large negative, so that it bias the overall likelihood.

### Why?

### Solution...



# Implementation Detail

### The integrated transpose operation in Cusparse MM is very expensive. Do it you self.