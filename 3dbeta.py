import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

class Environment:
    def __init__(self, size, num_plants):
        self.size = size
        self.map = np.random.rand(size, size, size) * 100  # 3D Environment values
        self.plants = np.hstack([np.random.rand(num_plants, 3) * size, np.random.rand(num_plants, 1) * 50])  # Plant positions and energy
        self.wind = np.random.rand(3) * 10  # Wind vector

class Animal:
    def __init__(self, environment, name, model=None):
        self.environment = environment
        self.name = name
        self.position = np.random.rand(3) * environment.size  # 3D position
        self.energy = 100
        if model is None:
            self.model = SGDRegressor(max_iter=1000, tol=1e-3)
        else:
            self.model = model
        self.scaler = StandardScaler().fit([self.get_state()])
        self.state = self.get_state()
        self.model.fit(self.scaler.transform([self.state]), [0])
        self.message = ""

    def get_state(self):
        return self.position.tolist() + [self.energy, self.energy, self.environment.map[int(self.position[0]), int(self.position[1]), int(self.position[2])]]

    def move(self):
        action = self.model.predict(self.scaler.transform([self.get_state()]))[0]
        self.position = (self.position + action) % self.environment.size  # Keep position within environment
        new_state = self.get_state()
        reward = new_state[-1] - self.state[-1]  # Reward is increase in environment value
        self.model.partial_fit(self.scaler.transform([self.state]), [reward])
        self.state = new_state
        self.energy -= 1

    def eat(self, plant_index):
        self.energy += self.environment.plants[plant_index, 3]
        self.environment.plants = np.delete(self.environment.plants, plant_index, axis=0)

    def attack(self, other):
        if self.energy > other.energy:
            self.energy += other.energy
            other.energy = 0
        else:
            other.energy += self.energy
            self.energy = 0

    def communicate(self):
        # If energy is low, communicate the nearest threat
        if self.energy < 30:
            nearest_animal = min((animal for animal in animals if animal is not self), key=lambda animal: np.linalg.norm(self.position - animal.position))
            self.message = f"Threat at {nearest_animal.position}"
        # Otherwise, communicate the most valuable known resource
        else:
            if len(self.environment.plants) > 0:
                most_valuable_plant = max(self.environment.plants, key=lambda plant: plant[3])
                self.message = f"Resource at {most_valuable_plant[:3]}"
            else:
                self.message = ""

    def reproduce(self):
        if self.energy >= 150:
            self.energy /= 2
            offspring_model = SGDRegressor(max_iter=1000, tol=1e-3)
            offspring_model.coef_ = self.model.coef_ + np.random.normal(scale=0.1, size=self.model.coef_.shape)  # Mutation
            offspring_model.intercept_ = self.model.intercept_ + np.random.normal(scale=0.1)  # Mutation
            offspring = Animal(self.environment, self.name + "'s offspring", model=offspring_model)
            offspring.position = self.position
            offspring.energy = self.energy
            return offspring
        else:
            return None

class Bird(Animal):
    def __init__(self, environment, name, model=None):
        super().__init__(environment, name, model)
        self.position = np.random.rand(3) * environment.size
        self.z = self.position[2]

    def get_state(self):
        return self.position.tolist() + [self.energy, self.energy, self.environment.map[int(self.position[0]), int(self.position[1]), int(self.position[2])]]

    def move(self):
        action = self.model.predict(self.scaler.transform([self.get_state()]))[0]
        self.position = (self.position + action + self.environment.wind) % self.environment.size  # Apply wind to movement

# Create an environment and some birds
environment = Environment(10, 20)
birds = [Bird(environment, f"Bird {i}") for i in range(10)]

# Each timestep, have each bird move, eat, communicate, and possibly reproduce
for t in range(10): # Limiting to 10 iterations for brevity
    print(f"--- Time step {t} ---")
    new_birds = []
    for i, bird in enumerate(birds):
        if bird.energy <= 0:
            continue
        bird.move()
        # If there's a plant nearby, eat it
        for j, plant in enumerate(environment.plants):
            if np.linalg.norm(bird.position - plant[:3]) < 1:
                bird.eat(j)
                break
        bird.communicate()
        offspring = bird.reproduce()
        if offspring is not None:
            new_birds.append(offspring)
    birds += new_birds
    # Remove dead birds
    birds = [bird for bird in birds if bird.energy > 0]

# Use matplotlib to visualize the final positions of the birds and plants
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for bird in birds:
    ax.scatter(*bird.position, color='red')
for plant in environment.plants:
    ax.scatter(*plant[:3], color='green')
plt.title("Final positions of birds (red) and plants (green) in the environment")
plt.show()
