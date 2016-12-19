from TrafficSignClassifier import TrafficSignClassifier


def main():

    tsc = TrafficSignClassifier('traffic-signs-data')
    tsc.generate_additional_training_features()
    tsc.train()

if __name__ == "__main__":
    main()