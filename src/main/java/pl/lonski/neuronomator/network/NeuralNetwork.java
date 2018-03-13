package pl.lonski.neuronomator.network;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Stream;


public class NeuralNetwork {

	private List<Layer> layers;
	private List<Synapse> synapses;
	private List<Double> expectedOutput;
	private DoubleUnaryOperator activationFn;
	private double totalError;
	private double toleratedError;

	public NeuralNetwork(NeuralNetworkParameters params) {
		this.expectedOutput = params.getExpectedOutput();
		this.activationFn = params.getActivationFn();
		this.toleratedError = params.getToleratedError();
		this.layers = new ArrayList<>();
		this.synapses = new ArrayList<>();
		createLayers(params);
	}

	private void createLayers(NeuralNetworkParameters params) {
		addLayer(new Layer(params.getInput()));
		Stream.generate(params::getHiddenLayerNeuronsCount)
				.map(Layer::new)
				.limit(params.getHiddenLayersCount())
				.forEach(this::addLayer);
		addLayer(new Layer(params.getExpectedOutput().size()));
	}

	private void addLayer(Layer layer) {
		layers.add(layer);
		if (layers.size() > 1) {
			Layer previousLayer = layers.get(layers.size() - 2);
			connectLayers(previousLayer, layer);
		}
	}

	private void connectLayers(Layer l1, Layer l2) {
		Random random = new SecureRandom();
		for (Neuron left : l1.getNeurons()) {
			for (Neuron right : l2.getNeurons()) {
				synapses.add(new Synapse(left, right, random.nextDouble()));
			}
		}
	}

	public double getTotalError() {
		return totalError;
	}

	public int getLayersCount() {
		return layers.size();
	}

	public void learn(int iterationCount) {
		for (int i = 0; i < iterationCount; ++i) {
			calculateNeuronValues();
			calculateError();
			System.out.print(String.format("\rIteration %d/%d : error = %f", i + 1, iterationCount, totalError));
			if (totalError < toleratedError) {
				System.out.println(String.format("Total error within limit: %f < %f", totalError, toleratedError));
				break;
			}
			recalculateOutputLayerWeights();
			recalculateHiddenLayerWeights();
			applyNewWeights();
		}
		System.out.println();
	}

	private void calculateNeuronValues() {
		for (int i = 1; i < layers.size(); i++) {
			calculateNeuronValues(layers.get(i), layers.get(i - 1));
		}
	}

	private void calculateNeuronValues(Layer layerToCalculate, Layer previousLayer) {
		layerToCalculate
				.modifyNeuronsData(right ->
						previousLayer.getNeurons().stream()
								.mapToDouble(left -> left.data * getSynapseWeight(left, right))
								.sum())
				.modifyNeuronsData(n -> activationFn.applyAsDouble(n.data));
	}

	private void calculateError() {
		Layer outputLayer = getOutputLayer();
		totalError = Stream.iterate(0, i -> i + 1)
				.mapToDouble(i -> expectedOutput.get(i) - outputLayer.getNeuronData(i))
				.map(diff -> 0.5 * diff * diff)
				.limit(outputLayer.getNeurons().size())
				.sum();
	}

	private void recalculateOutputLayerWeights() {
		Layer outputLayer = getOutputLayer();
		Layer leftLayer = layers.get(layers.size() - 2);
		for (int i = 0; i < outputLayer.getNeurons().size(); ++i) {
			Neuron right = outputLayer.getNeuron(i);
			right.dE_dOut = right.data - expectedOutput.get(i);
			right.dOut_dNet = right.data * (1 - right.data);
			calculateNewWeights(leftLayer, right);
		}
	}

	private void recalculateHiddenLayerWeights() {
		int lastHiddenLayerIdx = layers.size() - 2;
		for (int i = lastHiddenLayerIdx; i > 0; --i) {
			recalculateHiddenLayerWeights(layers.get(i - 1), layers.get(i), layers.get(i + 1));
		}
	}

	private void recalculateHiddenLayerWeights(Layer leftLayer, Layer currentLayer, Layer rightLayer) {
		for (Neuron current : currentLayer.getNeurons()) {
			current.dOut_dNet = current.data * (1 - current.data);
			current.dE_dOut = rightLayer.getNeurons().stream()
					.mapToDouble(right -> right.dE_dOut * right.dOut_dNet * getSynapseWeight(current, right))
					.sum();
			calculateNewWeights(leftLayer, current);
		}
	}

	private void calculateNewWeights(Layer leftLayer, Neuron right) {
		for (Neuron left : leftLayer.getNeurons()) {
			double dE_dW = right.dE_dOut * right.dOut_dNet * left.data;
			findSynapse(left, right).setWeightModificationFn(weight -> weight - 0.5 * dE_dW);
		}
	}

	private void applyNewWeights() {
		synapses.forEach(Synapse::modifyWeight);
	}

	private Synapse findSynapse(Neuron left, Neuron right) {
		return synapses.stream()
				.filter(s -> s.getLeft() == left && s.getRight() == right)
				.findFirst()
				.orElseThrow(MissingSynapseException::new);
	}

	private double getSynapseWeight(Neuron left, Neuron right) {
		return findSynapse(left, right).getWeight();
	}

	private Layer getOutputLayer() {
		return layers.get(layers.size() - 1);
	}

	private class MissingSynapseException extends RuntimeException {
	}
}
