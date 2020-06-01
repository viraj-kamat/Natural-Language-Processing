The advanved model is a Convolutional Neural model with Dense layers for class prediction :
1. We have a sequential model and add a single convolutional layer to it
2. We have maxpooling that follows this layer
3. We also add a dropout which random turns off some neurns in the model, to prevent overfitting
4. We add dense layers at the end (softmax classifier) returns the probalities of relations