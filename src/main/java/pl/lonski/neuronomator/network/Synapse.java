package pl.lonski.neuronomator.network;

import java.util.function.Function;

class Synapse {

	private Neuron left;
	private Neuron right;
	private double weight;
	private Function<Double, Double> weightModificationFn;

	Synapse(Neuron left, Neuron right, double weight) {
		this.left = left;
		this.right = right;
		this.weight = weight;
	}

	Neuron getLeft() {
		return left;
	}

	Neuron getRight() {
		return right;
	}

	double getWeight() {
		return weight;
	}

	void setWeightModificationFn(Function<Double, Double> weightModFn) {
		this.weightModificationFn = weightModFn;
	}

	void modifyWeight() {
		weight = weightModificationFn.apply(weight);
	}

}
