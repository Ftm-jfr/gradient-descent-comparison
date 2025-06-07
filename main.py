import time
import numpy as np
import matplotlib.pyplot as plt

# creating dataset
np.random.seed(42)
num_samples = 1000
x = np.linspace(0, 10, num_samples)
noise = np.random.normal(0, 0.5, num_samples)  # adding noise
y = 2 * x + 5 + noise


def create_batches(x, y, batch_size):
    indices = np.random.permutation(len(x))
    return [(x[indices[i:i + batch_size]], y[indices[i:i + batch_size]])
            for i in range(0, len(x), batch_size)]


def predict(x, w, b):
    return w * x + b


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def batch_gradient_step(xb, yb, w, b, lr):
    n = len(xb)
    y_pred = predict(xb, w, b)
    dw = (-2 / n) * np.sum(xb * (yb - y_pred))
    db = (-2 / n) * np.sum(yb - y_pred)
    w -= lr * dw
    b -= lr * db
    return w, b


def stochastic_gradient_descent(x, y, lr=0.01, epochs=100):
    w = 0
    b = 0

    loss_history = []
    path = []
    update_count = 0

    for epoch in range(epochs):
        idx = np.random.randint(len(x))
        xi = x[idx]
        yi = y[idx]

        y_pred = w * xi + b

        error = y_pred - yi
        dw = error * xi
        db = error

        w -= lr * dw
        b -= lr * db
        update_count += 1

        path.append((w, b))
        loss = np.mean((w * x + b - y) ** 2)
        loss_history.append(loss)

    return w, b, loss_history, path, update_count


def batch_gradient_descent(x, y, lr=0.01, epochs=100):
    w, b = 0.0, 0.0
    loss_history, path = [], []
    update_count = 0
    for epoch in range(epochs):
        w, b = batch_gradient_step(x, y, w, b, lr)
        update_count += 1
        y_pred = predict(x, w, b)
        loss_history.append(mean_squared_error(y, y_pred))
        path.append((w, b))

    return w, b, loss_history, path, update_count


def mini_batch_gradient_descent(x, y, lr=0.01, epochs=100, batch_size=32):
    w, b = 0.0, 0.0
    loss_history, path = [], []
    update_count = 0

    for epoch in range(epochs):
        batches = create_batches(x, y, batch_size)
        for xb, yb in batches:
            w, b = batch_gradient_step(xb, yb, w, b, lr)
            update_count += 1
            path.append((w, b))

        y_pred = predict(x, w, b)
        loss_history.append(mean_squared_error(y, y_pred))

    return w, b, loss_history, path, update_count


def plot_loss_comparison(loss_history_batch, loss_history_sgd, loss_history_mini_batch):
    plt.plot(loss_history_batch, label='Batch Gradient Descent', color='blue')
    plt.plot(loss_history_sgd, label='Stochastic Gradient Descent', color='green')
    plt.plot(loss_history_mini_batch, label='Mini-Batch Gradient Descent', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Function Comparison')
    plt.legend()
    plt.show()


def plot_contour_path(path_batch, path_sgd, path_mini_batch, w_range, b_range):
    W, B = np.meshgrid(w_range, b_range)
    Z = np.array([mean_squared_error(y, predict(x, w, b)) for w, b in zip(W.flatten(), B.flatten())])
    Z = Z.reshape(W.shape)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Batch GD
    axes[0].contour(W, B, Z, levels=50, cmap='jet')
    path_batch = np.array(path_batch)
    axes[0].plot(path_batch[:, 0], path_batch[:, 1], label='Batch GD Path', color='blue')
    axes[0].set_title('Batch Gradient Descent')
    axes[0].set_xlabel('w')
    axes[0].set_ylabel('b')
    axes[0].legend()

    # SGD
    axes[1].contour(W, B, Z, levels=50, cmap='jet')
    path_sgd = np.array(path_sgd)
    axes[1].plot(path_sgd[:, 0], path_sgd[:, 1], label='SGD Path', color='green')
    axes[1].set_title('Stochastic Gradient Descent')
    axes[1].set_xlabel('w')
    axes[1].set_ylabel('b')
    axes[1].legend()

    # Mini-Batch GD
    axes[2].contour(W, B, Z, levels=50, cmap='jet')
    path_mini_batch = np.array(path_mini_batch)
    axes[2].plot(path_mini_batch[:, 0], path_mini_batch[:, 1], label='Mini-Batch GD Path', color='red')
    axes[2].set_title('Mini-Batch Gradient Descent')
    axes[2].set_xlabel('w')
    axes[2].set_ylabel('b')
    axes[2].legend()

    plt.tight_layout()
    plt.show()


# comparing updates and time
start_time = time.time()
w_batch, b_batch, loss_history_batch, path_batch, update_count_batch = batch_gradient_descent(x, y, lr=0.01, epochs=3200)
end_time = time.time()
batch_time = end_time - start_time

start_time = time.time()
w_sgd, b_sgd, loss_history_sgd, path_sgd, update_count_sgd = stochastic_gradient_descent(x, y, lr=0.01, epochs=3200)
end_time = time.time()
sgd_time = end_time - start_time

start_time = time.time()
w_mini_batch, b_mini_batch, loss_history_mini_batch, path_mini_batch, update_count_mini_batch = mini_batch_gradient_descent(
    x, y, lr=0.01, epochs=100)
end_time = time.time()
mini_batch_time = end_time - start_time

# printing results
print(f"Batch GD - Time: {batch_time:.4f} seconds, Updates: {update_count_batch}")
print(f"SGD - Time: {sgd_time:.4f} seconds, Updates: {update_count_sgd}")
print(f"Mini-Batch GD - Time: {mini_batch_time:.4f} seconds, Updates: {update_count_mini_batch}")

# loss function comparison
plot_loss_comparison(loss_history_batch, loss_history_sgd, loss_history_mini_batch)

# contour plotting
w_range = np.linspace(-5, 5, 100)
b_range = np.linspace(-5, 5, 100)
plot_contour_path(path_batch, path_sgd, path_mini_batch, w_range, b_range)
