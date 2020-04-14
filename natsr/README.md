# NatSR

## Difference

1. Original repo (they) saves `already augmented (rot 90, 180, 270 degrees)` images. But, i just `run-time augmentation` w/ same methods.
2. They calculate `tv loss(?)`, then, save images selectively. But my implementation doesn't consider it.

## Proposed Architecture

### Explicitly Modeling the SISR & Designing Natural Manifold

![img](/assets/designing_natural_manifold.png)

As tons of `high-resolution` images can be correspond to one `low-resolution` image, so SISR task is *one-to-many problem*.

They define & separate `high-resolution space` into the `3 subspaces`, *blurry space*, *natural space*, *noisy space*.

### Natural Manifold Discrimination (NMD)

![img](/assets/nmd_architecture.png)
![img](/assets/nmd_loss.png)

Its' loss approach is like MinMax problem. Minimizing `unnatural manifold`, Maximizing `natural manifold` log-likely hood.

### Natural & Realistic SISR (NatSR)

![img](/assets/natsr_architecture.png)

![img](/assets/overall_architecture.png)

## Experimental Results

### FR-IQA Results

![img](/assets/fr-iqa-results.png)

## Reference

* Official Repository : [repo](https://github.com/JWSoh/NatSR/blob/master/README.md)
