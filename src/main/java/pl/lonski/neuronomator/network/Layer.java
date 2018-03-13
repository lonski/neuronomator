package pl.lonski.neuronomator.network;

import static java.util.stream.Collectors.toList;

import java.util.Collections;
import java.util.List;
import java.util.function.Function;

class Layer {

	private List<Neuron> neurons;

	Layer(List<Double> neuron_values) {
		this.neurons = neuron_values.stream()
				.map(Neuron::new)
				.collect(toList());
	}

	Layer(int neuronCount) {
		this(Collections.nCopies(neuronCount, 0.0));
	}

	List<Neuron> getNeurons() {
		return neurons;
	}

	Neuron getNeuron(int idx) {
		return neurons.get(idx);
	}

	double getNeuronData(int idx) {
		return getNeuron(idx).data;
	}

	Layer modifyNeuronsData(Function<Neuron, Double> newDataSupplier) {
		neurons.forEach(n -> n.data = newDataSupplier.apply(n));
		return this;
	}
}
