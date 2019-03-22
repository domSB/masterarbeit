import tensorflow as tf
import numpy as np
from collections import deque
import pickle

class StockSimulation:
    def __init__(self, sales, start_stock):
        assert type(sales) == np.ndarray, "Wrong type for sales"
        assert type(start_stock) == np.ndarray, "Wrong type for start_stock"
        self.sales = sales
        self.start_stock = start_stock
        self.days = len(sales)
        self.current_day = 0
        self.sim_stock = start_stock.copy()
        self.product_count = len(start_stock)

    def reset(self):
        self.sim_stock = self.start_stock.copy()
        self.current_day = 0
        self.sales_forecast = deque(maxlen=4)
        for i in range(4):
            self.sales_forecast.append(sales[i])
        new_state = np.append(self.start_stock, self.sales_forecast).reshape(5, 5)
        return new_state

    def make_action(self, action):
        assert len(action) == self.product_count, "len(actions) doesn't match lenght of product stock"
        assert self.current_day < self.days - 1 , "Episode is finished. Do Reset."
        action = np.array(action).astype(np.uint8)
        reward = 0.0
        self.sim_stock -= sales[self.current_day]
        for stock in self.sim_stock:
            if stock < 0:
                reward -= (1/self.product_count)
        inventory = np.sum(self.sim_stock.clip(0))
        if inventory != 0:
            reward += 1 / inventory
        self.sim_stock += action
        self.current_day += 1
        if self.current_day + 4 < self.days:
            self.sales_forecast.append(sales[self.current_day + 4])
        else:
            self.sales_forecast.append(np.zeros(5).astype(int))
        new_state = np.append(self.sim_stock, self.sales_forecast).reshape(5, 5)
        return reward, self.current_day == self.days - 1, new_state


with open("./data/sales.pickle", "rb") as file:
    sales = pickle.load(file)

with open("./data/inventory.pickle", "rb") as file:
    start_stock = pickle.load(file)

simulation = StockSimulation(sales, start_stock)



