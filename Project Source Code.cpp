#include<bits/stdc++.h>
#define LL long long int
#define maxn 100
using namespace std;
//--------------------program parameters---------------------
int TESTSIZE,TRAINSIZE,ITER; 		//40
const char* train_file="training-set.txt";
const char* test_file ="test-set2.txt";
const char* output_file="output2.txt";
//const char* input_file="input.txt";
double INITIAL_WEIGHTS[] = {-0.996246, 0.690756, -0.420087, 1.42622, 0.755028, 0.439619, 0.0508744 ,1.68789 ,1.46852 ,1.23981 ,-0.477676 
,1.57683, 1.1315, 0.540605 ,-0.0880154, -0.955046 ,-0.725791 ,0.0933561 ,-0.558061 ,-0.502304 ,1.96558 ,0.337077 ,-0.64275 ,-0.985992 
,-0.973266, 0.133641, 0.594989 ,0.713553, 0.805292, 0.821497 ,-0.501297, 0.989136 ,0.352367 ,0.0563678 ,-0.828883 }; //int j,k;
//--------Neuron and Layers(Basic Structures)------------------
// things required = data,weights,error,dw
struct Neuron{
		double data; // data
		double error; /*error*/
		double* w;
		double* dw;//weights and change in weights dw
};
struct Layers{
	LL neuron_num; //number of neurons in a layer
	Neuron* neuron; // neurons in the layer 
};

//------------MLP-----
class MLP{
	int lnum; // number of layers
	double eta;
	Layers* layers;
	public:
		MLP(LL number_of_layers,LL number_of_neurons[]):
			lnum(number_of_layers),
			layers(0),
			eta(0.009) // let eta be 0.5
		{
			//cout<<"problem"<<endl;
			layers = new Layers[number_of_layers];
			// for each layer
			for(int i=0;i<number_of_layers;i++){
				
				layers[i].neuron_num = number_of_neurons[i];
				layers[i].neuron = new Neuron[number_of_neurons[i]];
				// for each neuron
				for(int j=0;j<number_of_neurons[i];j++){
					layers[i].neuron[j].data = 1.0; // let initial data be 1.0 
					layers[i].neuron[j].error = 0.0;  
				  	if(i>0){
				  		// number of weights = number of neurons in previous layer
						layers[i].neuron[j].w = new double[ number_of_neurons[i-1] ];
						layers[i].neuron[j].dw = new double[ number_of_neurons[i-1] ];
						  
					}
					else{
						layers[i].neuron[j].w = 0;
						layers[i].neuron[j].dw = 0;	
					
					}
				}
			}
		}
		~MLP(){
			for(int i=0; i<lnum; ++i){
	    		if (layers[i].neuron){
	  				for(int j=0; j<layers[i].neuron_num; ++j){
	      				if (layers[i].neuron[j].w)
							delete [] layers[i].neuron[j].w;
	    			}
				}
	    		delete [] layers[i].neuron;
			}
			delete [] layers;
		}
		void setinput(double* input){
			//input is only for the first layer
			for(int i=0;i<layers[0].neuron_num;i++){
				layers[0].neuron[i].data = input[i];
				//cout<<layers[0].neuron[i].data<<" ";
			}
			//cout<<endl;
		}
		void initialise_weights(double weights[]){
			// not for first layer obviously
			int n = 0;
			for(int i=1;i<lnum;i++){
				for(int j=0;j<layers[i].neuron_num;j++){
					for(int k=0;k<layers[i-1].neuron_num;k++,n++){
						layers[i].neuron[j].w[k] = weights[n];
					}
				}
			}
		}
		void initialise_dw(){
			for(int i=1;i<lnum;i++){
				for(int j=0;j<layers[i].neuron_num;++j){
					for(int k=0;k<layers[i-1].neuron_num;++k){
						layers[i].neuron[j].dw[k]=0;
					}
					
				}
			}
			//cout<<"reaching the end"<<endl;
		}
		void getoutput(double* output){
			//last layer
			for(int j=0;j<layers[lnum-1].neuron_num;j++){
				output[j] = layers[lnum-1].neuron[j].data;
			}
		}
		void forwardpropagate(){
			//every layer except the first one
			for(int i=1;i<lnum;i++){
				//for every neuron in the layer
				for(int j=0;j<layers[i].neuron_num;j++){
					//get input from last layer and calculate net
					double net = 0.0;
					//for each output from last layer
					for(int k=0;k<layers[i-1].neuron_num;k++){
						double o = layers[i-1].neuron[k].data;
						double w = layers[i].neuron[j].w[k] ; //weights from k to j
						net += o*w;
						//cout<<o<<" "<<w<<endl;
					}
					//cout<<net<<endl;
					//activate
					layers[i].neuron[j].data = 1/(1+exp(-net));
					//cout.precision(8);
					//printf("%llf\n",log(1+(double)exp(-net)));
					//cout<<layers[i].neuron[j].data<<endl;
				}
			}
		}
		//for each output unit k do dk ? ok(1 - ok)(tk - ok)
		void computeerror(double* target){
			//last layer
			
			for(int j=0; j<layers[lnum-1].neuron_num; ++j){
				double o = layers[lnum-1].neuron[j].data;
				double t = target[j];
				//cout<<o<<" "<<t<<endl;
				// cout<<"target["<<j<<"] is "<<t<<" with output "<<o<<endl;
				layers[lnum-1].neuron[j].error = o*(1-o)*(t-o);
			//	cout<<o*(1-o)*(t-o)<<endl;
			}
			
		}
		void backpropagate(){
			//from second last layer
			for(int i=lnum-2;i>=0;i--){
				// for each hidden layer
				for(int j=0;j<layers[i].neuron_num;j++){
					double o = layers[i].neuron[j].data;
					double E = 0.0;
					//get w and e from the upper layer, compute sigma w*e
					for(int k=0;k<layers[i+1].neuron_num;k++){
						E+=layers[i+1].neuron[k].w[j]*layers[i+1].neuron[k].error;
					}
					layers[i].neuron[j].error = o*(1-o)*E;
					//cout<<layers[i].neuron[j].error<<endl;
				}
			}
		}
		void updatedw(){
			// ignore first layer
			for(int i=1;i<lnum;i++){
				for(int j=0;j<layers[i].neuron_num;j++){
					for(int k=0;k<layers[i-1].neuron_num;k++){
						layers[i].neuron[j].dw[k] += eta*layers[i].neuron[j].error*layers[i-1].neuron[k].data;
					}
				}
			}
			//cout<<"reaching the end"<<endl;
		}
		void updateWeights(){
			// ignore first layer
			for(int i=1;i<lnum;i++){
				// for each neuron in the layer
				for(int j=0;j<layers[i].neuron_num;j++){
					// each weight from the last layer
					for(int k=0;k<layers[i-1].neuron_num;k++){
						layers[i].neuron[j].w[k] += layers[i].neuron[j].dw[k];
					}
				}
			}
		}
		void batch_train(double* input, double* target,int size, double* output=0){
			initialise_dw();
			int instep = layers[0].neuron_num;
			int outstep = layers[lnum -1 ].neuron_num;
			//for each example
			for(int i=0;i<size;i++){
				setinput(input+i*instep);
				forwardpropagate();
				computeerror(target+i*outstep);
				backpropagate();
				updatedw();
			}
			updateWeights();
			if(output) getoutput(output);
		//cout<<"reaching the end"<<endl;
		}
		void Train(double* input,double* weights,double* target,int size,int it,double* records=0,double* output=0){
			//cout<<"problem";
			initialise_weights(weights);
			for(int i=0;i<it;i++){
				for(int j=0;j<10;j++){
					batch_train(input,target,size,output);
					//cout<<"calling batchtrain"<<endl;
				}
				
			}
			
		}
		
		void Test(double* input,double* output,int size){
			for(int i=0;i<size;i++){
				setinput(input+6*i);
				forwardpropagate();
				getoutput(output+i);
			}
		}
};
//------------------data processing------------------------------
void loaddata(double data[],const char* file){
	double n;
	int i=0;
	ifstream fin(file);
	//fin.open(file,ios::in);
	while(fin>>n){
		data[i++] = n;
		//cout<<data[i-1]<<" ";
	}
	//cout<<"problem"<<endl;
	/*for(i=0;i<6*TRAINSIZE;i++)
		cout<<data[i]<<" ";
	*/fin.close();
}
void GenTrainSet(double raw[], double In[],double Target[]){
 for(int i=0;i<TRAINSIZE;++i){
		In[6*i]       =1.0;
		In[6*i+1]     =raw[6*i];
		In[6*i+2]	  =raw[6*i+1];
		In[6*i+3]	  =raw[6*i+2]; 	
		In[6*i+4]	  = raw[6*i+3];
		In[6*i+5] 	  = raw[6*i+4];
		Target[i]     =raw[6*i+5];
		//cout<<In[6*i]<<" "<<In[6*i+1]<<" "<<In[6*i+2]<<" "<<In[6*i+3]<<" "<<In[6*i+4]<<" "<<In[6*i+5]<<" "<<Target[i]<<endl;
 	//cout<<"problem"<<endl;
 }
}

void GenInput(double* raw, double* Input){
	for(int i=0;i<TESTSIZE;++i){
		Input[6*i]        =1.0;
		Input[6*i+1]      =raw[5*i];
		Input[6*i+2]	  =raw[5*i+1];
		Input[6*i+3]	  =raw[5*i+2]; 	
		Input[6*i+4]	  =raw[5*i+3];
		Input[6*i+5]	  =raw[5*i+4];
		//cout<<Input[6*i+1]<<" "<<Input[6*i+2]<<" "<<Input[6*i+3]<<" "<<Input[6*i+4]<<" "<<Input[6*i+5]<<endl;
	}
}
//-----------------------main function---------------------------
int main(){
	/*LL layer[3]={6,5,1};*/
	int number_of_layers;
		
	fstream fin("input-file.txt",ios::in);
	fin>>TESTSIZE;
	fin>>TRAINSIZE;
	fin>>ITER;
	fin>>number_of_layers;
	LL layer[number_of_layers];
	for(int i=0;i<number_of_layers;i++)
		fin>>layer[i];
	MLP test(number_of_layers,layer);
	double raw_data[6*TRAINSIZE];
		//--------------------Train for ITER times--------------------
	double Input[6*TRAINSIZE];
	double Target[6*TRAINSIZE];
	double records[maxn];
	loaddata(raw_data,train_file);
	GenTrainSet(raw_data,Input,Target);
	test.Train(Input,INITIAL_WEIGHTS,Target,TRAINSIZE,ITER,records);
	//--------------------Test testcase--------------------
	double TestInput[5*TESTSIZE];
	double TestOutput[6*TESTSIZE];
	loaddata(raw_data,test_file);
	GenInput(raw_data,TestInput);
	test.Test(TestInput,TestOutput,10);
	ofstream myfile;
	myfile.open (output_file);
	//--------------------output result--------------------
	myfile<<"output=[";
	for(int i=0;i<TESTSIZE;++i){
		myfile<<TestOutput[i]<<endl;
	}
	myfile<<"]\n";
	fin.close();
  myfile.close();
	return 0;
}
