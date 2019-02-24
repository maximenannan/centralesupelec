# Deep learning

There were 3 assignments to achieve:

**MP1**:

It was about linear classifier, convolutional neural network and hourglass network. There were 3 
problems:
1) Given a set of images predicting whether it's a triangle, a rectangle or a circle.
2) Given a set of triangle, predicting its vertices.
3) Given a set of noisy images, denoising them. 

Results:
1) Accuracy of 94%
2) No results
3) Accuracy of 99.5%

**MP3**

It was about reinforcement learning and mainly Deep Q-learning:
```
A rat runs on an island and tries to eat as much as possible. The island is subdivided into $N\times N$ cells, in which there are cheese (+0.5) and poisonous cells (-1). The rat has a visibility of 2 cells (thus it can see $5^2$ cells). The rat is given a time $T$ to accumulate as much food as possible. It can perform 4 actions: going up, down, left, right. 

The goal is to code an agent to solve this task that will learn by trial and error. We propose the following environment:
 
