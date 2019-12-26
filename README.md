# Vanilla Policy Gradient with Lunar Lander
## Results
Agent before & after training:

![agent playing lunar lander before training](https://user-images.githubusercontent.com/54861081/71486048-38054280-27c9-11ea-984d-e3154fe22790.gif)
![agent playing lunar lander after training](https://user-images.githubusercontent.com/54861081/71486097-7d297480-27c9-11ea-826c-75ca853ca0cc.gif)

Training is very noisy:

![rewards vs episodes during training](https://user-images.githubusercontent.com/54861081/71485792-d7c1d100-27c7-11ea-98c4-5169543a9a59.png)
![steps vs episodes during training](https://user-images.githubusercontent.com/54861081/71485797-dd1f1b80-27c7-11ea-9d34-0e088ba3f049.png)
![avg reward vs iters during eval](https://user-images.githubusercontent.com/54861081/71485798-e1e3cf80-27c7-11ea-8573-5c8a97dc2e22.png)
![max reward vs episodes during training](https://user-images.githubusercontent.com/54861081/71485806-e4dec000-27c7-11ea-81bb-132ad12f287b.png)

## Todo
* batch episodes for training (currently updating policy after each episode)
* add baseline and other variance reducing techniques
* try different algos altogether 