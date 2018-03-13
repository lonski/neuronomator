package pl.lonski.neuronomator.network;

class Neuron {

	Neuron(double data) {
		this.data = data;
	}

	double data;
	double dE_dOut;
	double dOut_dNet;
}
