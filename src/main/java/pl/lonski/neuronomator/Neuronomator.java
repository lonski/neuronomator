package pl.lonski.neuronomator;

import static java.util.Arrays.asList;
import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;
import static pl.lonski.neuronomator.network.NeuralNetworkParameters.ActivationFunctions.LOGISTIC;

import java.util.List;
import java.util.Scanner;

import pl.lonski.neuronomator.network.NeuralNetwork;
import pl.lonski.neuronomator.network.NeuralNetworkParameters;
import pl.lonski.neuronomator.network.TopologyPrinter;

public class Neuronomator {

	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		NeuralNetworkParameters params = new NeuralNetworkParameters();
		boolean isAccepted = false;

		System.out.println(":: Neuronomator 6000 ::");
		System.out.println();

		while (!isAccepted) {
			printSeparator("Menu");
			System.out.println("1. Enter network parameters");
			System.out.println("2. Use example parameter set");
			System.out.println("0. Exit");
			System.out.print(">");

			switch (scanner.nextInt()) {
			case 1:
				params = promptUserForParams(scanner);
				break;
			case 2:
				params = getExampleParams();
				break;
			default:
				System.exit(0);
			}

			System.out.println(new TopologyPrinter().print(params));

			System.out.print("Ok? [y/n]>");
			scanner.nextLine();
			isAccepted = scanner.nextLine().equals("y");
		}

		System.out.print("Enter maximum number of iterations\n>");
		int iterations = scanner.nextInt();
		System.out.print("Enter tolerated error value\n>");
		params.setToleratedError(scanner.nextDouble());

		printSeparator("Learning");
		NeuralNetwork nn = new NeuralNetwork(params);
		nn.learn(iterations);

		printSeparator("Calculated weights");
		System.out.println(nn.printWeights());
	}

	private static NeuralNetworkParameters getExampleParams() {
		return new NeuralNetworkParameters()
				.setInput(asList(0.05, 0.1))
				.setExpectedOutput(asList(0.01, 0.99))
				.setHiddenLayersCount(2)
				.setHiddenLayerNeuronsCount(3)
				.setToleratedError(0.001)
				.setActivationFn(LOGISTIC);
	}

	private static NeuralNetworkParameters promptUserForParams(Scanner scanner) {
		NeuralNetworkParameters params = new NeuralNetworkParameters();
		params.setActivationFn(LOGISTIC);

		System.out.print("Enter neuron values of input layer (space separated numbers)\n>");
		scanner.nextLine();
		params.setInput(readList(scanner));
		System.out.print("Enter expected neuron values of output layer\n>");
		params.setExpectedOutput(readList(scanner));
		System.out.print("Enter number of hidden layers\n>");
		params.setHiddenLayersCount(scanner.nextInt());
		System.out.print("Enter amount of neurons in hidden layers\n>");
		params.setHiddenLayerNeuronsCount(scanner.nextInt());

		return params;
	}

	private static List<Double> readList(Scanner scanner) {
		return stream(scanner.nextLine().split(" ")).map(Double::valueOf).collect(toList());
	}

	private static void printSeparator(String msg) {
		msg = "[" + msg.replaceAll(" ", "~") + "]";
		String line = String.format("-----%-80s%n", msg)
				.replaceAll(" ", "-")
				.replaceAll("~", " ");
		System.out.print(line);

	}
}
