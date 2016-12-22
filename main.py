from TrafficSignClassifier import TrafficSignClassifier


def main():
    tsc = TrafficSignClassifier('traffic-signs-data')
    tsc.basic_summary()

    tsc.generate_additional_training_features()
    tsc.pre_process_features()
    tsc.generate_ohe_encoding()
    tsc.split_training_set()

    tsc.train(run_optimization=True, batch_size=2, training_epochs=1)

if __name__ == "__main__":
    main()