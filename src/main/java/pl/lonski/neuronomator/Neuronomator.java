package pl.lonski.neuronomator;

import static java.util.Arrays.asList;
import static pl.lonski.neuronomator.network.NeuralNetworkParameters.ActivationFunctions.LOGISTIC;

import pl.lonski.neuronomator.network.NeuralNetwork;
import pl.lonski.neuronomator.network.NeuralNetworkParameters;

public class Neuronomator {

	public static void main(String[] args) {
		NeuralNetworkParameters params = new NeuralNetworkParameters()
				.setInput(asList(0.05, 0.1))
				.setExpectedOutput(asList(0.01, 0.99))
				.setHiddenLayersCount(2)
				.setHiddenLayerNeuronsCount(3)
				.setToleratedError(0.001)
				.setActivationFn(LOGISTIC);

		NeuralNetwork nn = new NeuralNetwork(params);

		nn.learn(1000000);
	}
}
