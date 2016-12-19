from TrafficSignClassifier import TrafficSignClassifier


def main():

    tsc = TrafficSignClassifier('traffic-signs-data')
    tsc.basic_summary()
    tsc.generate_additional_training_features()
    tsc.basic_summary()
    #tsc.train()

if __name__ == "__main__":
    main()