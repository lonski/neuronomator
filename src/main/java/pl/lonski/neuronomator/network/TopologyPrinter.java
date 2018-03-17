package pl.lonski.neuronomator.network;

import java.util.Formatter;

public class TopologyPrinter {

	public String print(NeuralNetworkParameters params) {
		StringBuilder sb = new StringBuilder();
		Formatter f = new Formatter(sb);

		final int rowCount = Math.max(params.getHiddenLayerNeuronsCount(),
				Math.max(params.getInput().size(), params.getExpectedOutput().size())) + 1;

		final int iRowSkip = calculateRowSkip(rowCount, params.getInput().size());
		final int hRowSkip = calculateRowSkip(rowCount, params.getHiddenLayerNeuronsCount());
		final int oRowSkip = calculateRowSkip(rowCount, params.getExpectedOutput().size());

		f.format("%n");
		for (int row = 1; row < rowCount; row++) {
			//input
			if (shouldPrint(row, iRowSkip, params.getInput().size())) {
				f.format("|%6.3f| ", params.getInput().get(row - iRowSkip));
			} else {
				f.format("         ");
			}
			//hidden
			String hSymbol = shouldPrint(row, hRowSkip, params.getHiddenLayerNeuronsCount()) ? "|o| " : "    ";
			for (int hIdx = 0; hIdx < params.getHiddenLayersCount(); hIdx++) {
				f.format(hSymbol);
			}
			//output
			if (shouldPrint(row, oRowSkip, params.getExpectedOutput().size())) {
				f.format("|%6.3f| ", params.getExpectedOutput().get(row - oRowSkip));
			} else {
				f.format("         ");
			}
			f.format("%n");
		}
		return sb.toString();
	}

	private boolean shouldPrint(int currentRow, int rowSkip, int elementCount) {
		return currentRow >= rowSkip && currentRow - rowSkip < elementCount;
	}

	private int calculateRowSkip(int rowCount, int elementCount) {
		return Math.max(0, (rowCount - elementCount) / 2) + 1;
	}
}
