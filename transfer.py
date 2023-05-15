import numpy as np

if __name__ == "__main__":

    np.set_printoptions(suppress=True)
    data = np.load("Store_data.npy")
    label = np.load("Store_label.npy").reshape(-1,1)
    print(data.shape)
    print(label.shape)
    X = np.concatenate([data, label], axis=1)
    print(X.shape)
    np.savetxt("support_data.txt", X,  delimiter=',', fmt='%d')