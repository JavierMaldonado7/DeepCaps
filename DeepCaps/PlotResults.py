import matplotlib.pyplot as plt

def plot_loss(train_loss, test_loss):
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Capsule Network Loss')
