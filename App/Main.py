import Preprocessing as pr

if __name__ == "__main__":
    data_path = 'data/input/*.csv'
    data = pr.preprocessing(data_path=data_path)

    print(data)