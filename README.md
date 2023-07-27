# EcoSim
EcoSim is a Python-based simulation of an ecosystem with plants and animals, showcasing interactions such as hunting and communication. Animals use machine learning (SGDRegressor) to optimize their movements and interactions in response to the environment and communicate the locations of resources and threats

![EcoSim Plot](ecosim_plot.png.png "Final positions of animals and plants in the simulation")


# EcoSim 3d beta 


This script models an environment where "birds" and plants exist.

The environment is a 3D grid of a certain size, each point in the grid has a value, and this grid represents different "resources" in the environment. In this environment, there are plants placed randomly which are considered as resources for birds.

The birds in this environment are modeled as agents with a 3D position, a certain amount of energy, and a predictive model that dictates their behavior. The predictive model is a Stochastic Gradient Descent (SGD) Regressor, which is a type of machine learning model. This model is used to predict the next action the bird will take based on its current state.

The state of a bird is represented by its position in the 3D grid, its current energy level, and the environment map value at its position. The bird uses this state information to predict the next action it will take.

An action is represented by a movement in the 3D grid. After predicting an action, the bird moves in the environment. The new position is calculated based on the bird's current position, the predicted action, and the wind in the environment.

Birds can also "eat" plants if they are nearby, which increases their energy. They can also communicate with each other to alert about threats or valuable resources. If a bird's energy level reaches a certain threshold, it can reproduce, creating a new bird. The offspring will have similar (but slightly mutated) behavior as the parent.

The script runs in a loop where each iteration represents a timestep. At each timestep, all birds move, eat if they can, communicate, and potentially reproduce.

Finally, the script visualizes the final positions of the birds and plants in the 3D environment using matplotlib's 3D plotting capabilities. The birds are represented as red dots and the plants as green dots.

![EcoSim Plot](ecosim_plot.png.png "Final positions of animals and plants in the simulation")
