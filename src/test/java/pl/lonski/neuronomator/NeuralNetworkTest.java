package pl.lonski.neuronomator;

import static java.util.Arrays.asList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static pl.lonski.neuronomator.network.NeuralNetworkParameters.ActivationFunctions.LOGISTIC;

import org.junit.Test;

import pl.lonski.neuronomator.network.NeuralNetwork;
import pl.lonski.neuronomator.network.NeuralNetworkParameters;

public class NeuralNetworkTest {

	@Test
	public void shouldCreateLayers() {
		NeuralNetworkParameters params = getParams();
		NeuralNetwork nn = new NeuralNetwork(params);

		int layerCount = nn.getLayersCount();

		assertEquals(params.getHiddenLayersCount() + 2, layerCount);
	}

	@Test
	public void shouldReduceError() {
		NeuralNetwork nn = new NeuralNetwork(getParams());

		nn.learn(1);
		double initError = nn.getTotalError();
		nn.learn(100);
		double outError = nn.getTotalError();

		assertTrue(outError < initError);
	}

	NeuralNetworkParameters getParams() {
		return new NeuralNetworkParameters()
				.setInput(asList(0.05, 0.1))
				.setExpectedOutput(asList(0.01, 0.99))
				.setHiddenLayersCount(2)
				.setHiddenLayerNeuronsCount(3)
				.setToleratedError(0.001)
				.setActivationFn(LOGISTIC);
	}
}
