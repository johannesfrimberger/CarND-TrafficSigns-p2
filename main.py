from TrafficSignClassifier import TrafficSignClassifier


def main():
    tsc = TrafficSignClassifier('traffic-signs-data')
    tsc.basic_summary()

    tsc.generate_additional_training_features()
    tsc.pre_process_features()
    tsc.generate_ohe_encoding()
    tsc.split_training_set()

    tsc.train(run_optimization=False, batch_size=2, training_epochs=1, dropout=True, l2_reg=True)

    tsc.save_model("model.ckpt")

if __name__ == "__main__":
    main()