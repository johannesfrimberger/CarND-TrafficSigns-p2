from TrafficSignClassifier import TrafficSignClassifier


def main():
    tsc = TrafficSignClassifier('traffic-signs-data')
    tsc.basic_summary()

    #tsc.split_training_set()
    #tsc.pre_process_features()
    #tsc.generate_additional_training_features()
    #tsc.generate_ohe_encoding()

    #tsc.train(run_optimization=True, batch_size=2, training_epochs=1, dropout=False, l2_reg=False)

if __name__ == "__main__":
    main()