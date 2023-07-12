import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

class Environment:
    def __init__(self, size, num_plants):
        self.size = size
        self.map = np.random.rand(size, size) * 100  # Environment values
        self.plants = np.hstack([np.random.rand(num_plants, 2) * size, np.random.rand(num_plants, 1) * 50])  # Plant positions and energy

class Animal:
    def __init__(self, environment, name, model=None):
        self.environment = environment
        self.name = name
        self.position = np.random.rand(2) * environment.size
        self.energy = 100
        if model is None:
            self.model = SGDRegressor(max_iter=1000, tol=1e-3)
        else:
            self.model = model
        self.scaler = StandardScaler().fit([self.position.tolist() + [0, 0]]) # Corrected the initial fit of the scaler
        self.state = self.get_state()
        self.model.fit(self.scaler.transform([self.state]), [0]) # Initial fit of the model
        self.message = ""

    def get_state(self):
        return self.position.tolist() + [self.energy] + [self.environment.map[int(self.position[0]), int(self.position[1])]]

    def move(self):
        action = self.model.predict(self.scaler.transform([self.get_state()]))[0]
        self.position = (self.position + action) % self.environment.size  # Keep position within environment
        new_state = self.get_state()
        reward = new_state[-1] - self.state[-1]  # Reward is increase in environment value
        self.model.partial_fit(self.scaler.transform([self.state]), [reward])
        self.state = new_state
        self.energy -= 1

    def eat(self, plant_index):
        self.energy += self.environment.plants[plant_index, 2]
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
                most_valuable_plant = max(self.environment.plants, key=lambda plant: plant[2])
                self.message = f"Resource at {most_valuable_plant[:2]}"
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

# Create an environment and some animals
environment = Environment(10, 20)
animals = [Animal(environment, f"Animal {i}") for i in range(10)]

# Each timestep, have each animal move, eat, or attack, communicate, and possibly reproduce
for t in range(10): # Limiting to 10 iterations for brevity
    print(f"--- Time step {t} ---")
    new_animals = []
    for i, animal in enumerate(animals):
        if animal.energy <= 0:
            continue
        animal.move()
        # If there's a plant nearby, eat it
        for j, plant in enumerate(environment.plants):
            if np.linalg.norm(animal.position - plant[:2]) < 1:
                animal.eat(j)
                break
        # Otherwise, if there's another animal nearby, attack it
        else:
            for j, other in enumerate(animals):
                if i != j and np.linalg.norm(animal.position - other.position) < 1:
                    animal.attack(other)
                    break
        animal.communicate()
        offspring = animal.reproduce()
        if offspring is not None:
            new_animals.append(offspring)
    animals += new_animals
    # Remove dead animals

animals = [animal for animal in animals if animal.energy > 0]


    animals = [animal for animal in animals if animal.energy > 0]

# Use matplotlib to visualize the final positions of the animals and plants
plt.figure()
for animal in animals:
    plt.plot(*animal.position, 'ro')
for plant in environment.plants:
    plt.plot(*plant[:2], 'go')
plt.title("Final positions of animals (red) and plants (green) in the environment")
plt.show()
