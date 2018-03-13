package pl.lonski.neuronomator.network;

import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;

public class NeuralNetworkParameters {

	private List<Double> input;
	private List<Double> expectedOutput;
	private int hiddenLayersCount;
	private int hiddenLayerNeuronsCount;
	private DoubleUnaryOperator activationFn;
	private double toleratedError;

	public List<Double> getInput() {
		return input;
	}

	public NeuralNetworkParameters setInput(List<Double> input) {
		this.input = input;
		return this;
	}

	public List<Double> getExpectedOutput() {
		return expectedOutput;
	}

	public NeuralNetworkParameters setExpectedOutput(List<Double> expectedOutput) {
		this.expectedOutput = expectedOutput;
		return this;
	}

	public int getHiddenLayersCount() {
		return hiddenLayersCount;
	}

	public NeuralNetworkParameters setHiddenLayersCount(int hiddenLayersCount) {
		this.hiddenLayersCount = hiddenLayersCount;
		return this;
	}

	public int getHiddenLayerNeuronsCount() {
		return hiddenLayerNeuronsCount;
	}

	public NeuralNetworkParameters setHiddenLayerNeuronsCount(int hiddenLayerNeuronsCount) {
		this.hiddenLayerNeuronsCount = hiddenLayerNeuronsCount;
		return this;
	}

	public DoubleUnaryOperator getActivationFn() {
		return activationFn;
	}

	public NeuralNetworkParameters setActivationFn(DoubleUnaryOperator activationFn) {
		this.activationFn = activationFn;
		return this;
	}

	public double getToleratedError() {
		return toleratedError;
	}

	public NeuralNetworkParameters setToleratedError(double toleratedError) {
		this.toleratedError = toleratedError;
		return this;
	}

	public interface ActivationFunctions {
		DoubleUnaryOperator LOGISTIC = x -> 1.0 / (1.0 + Math.exp(-1.0 * x));
	}
}
