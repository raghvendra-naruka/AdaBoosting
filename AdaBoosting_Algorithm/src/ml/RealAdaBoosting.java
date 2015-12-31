package ml;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

//Real AdaBoosting
public class RealAdaBoosting {
	int totalInstances;
	double epsilon;
	List<Instance> instanceList;
	List<Classification> classifierSet;
	List<Classification> selectedClassifier;

	public RealAdaBoosting(List<Instance> instanceList, List<Classification> hypothesis, int totalItrs, int totalInstances, double epsilon) {
		this.instanceList = instanceList;
		this.classifierSet = hypothesis;
		this.selectedClassifier = new LinkedList<>();
		this.totalInstances = totalInstances;
		this.epsilon = epsilon;
	}

	//choosing optimum classifier
	public Classification chooseBestClassifier() {
		Classification bestClassifier = null;
		double minG = Integer.MAX_VALUE;
		for(int i = 0 ; i < classifierSet.size(); i++){
			Classification classifier = classifierSet.get(i);
			classifier.computeProbabilities(instanceList);
			classifier.computeG();
			if(classifier.G < minG ) {
				minG = classifier.G;
				bestClassifier = classifier;
			}
		}
		return bestClassifier;
	}

	//updating new probabilities 
	private double updateProbabilities(Classification classifier) {
		double normalizationFactor = 0, q;
		for(int i = 0; i < instanceList.size(); i++){	
			Instance instance = instanceList.get(i);
			if(classifier.classifiedInstances.get(i).getTagetAttr () == 1) {
				q = Math.pow(Math.E, -1 * instance.getTagetAttr () * classifier.cPlus );
			} else {
				q = Math.pow(Math.E, -1 * instance.getTagetAttr () * classifier.cMinus );
			}
			instance.setPrb (instance.getPrb () * q );
			normalizationFactor += instance.getPrb ();
		}	

		for(int i = 0; i < instanceList.size(); i++) {
			instanceList.get(i).setPrb (instanceList.get(i).getPrb () / normalizationFactor);
		}
		return normalizationFactor;
	}

	//target attribute computation
	private double computeTargetAttribute(int index) {
		double targetAttribute = 0;
		for(Classification classifier : selectedClassifier) {
			if(classifier.classifiedInstances.get(index).getTagetAttr () == 1) {
				targetAttribute +=  classifier.cPlus;
			}
			else {
				targetAttribute +=  classifier.cMinus;
			}
		}		
		return targetAttribute;
	}

	//real adaboosting calculations
	public void performRealAdaBoost (int totalItrs) throws FileNotFoundException {
		double e = 2.7182818284;
		ReadFile Rf = new ReadFile("adaboost-5.txt");
		ArrayList<Double> flist = new ArrayList<>();
		double bound = 1.0;
		for (int i = 0; i < ReadFile.n; i++) {
			flist.add(0.0);
		}

		for (int l = 0; l < ReadFile.T; l++) {
			System.out.println("\nIteration " + (l+1));
			double minG = 500.0;
			int x_pos = 0;
			boolean flag = true;
			double alpha = 0.0;
			double z = 0;
			double final_prp=0.0,final_prn=0.0,final_pwp=0.0,final_pwn=0.0;

			for (int i = 0; i < ReadFile.n - 1; i++) {
				double pr_p = 0.0, pr_n = 0.0, pw_p = 0.0, pw_n = 0.0;
				for (int j = 0; j <= i; j++) {
					if (ReadFile.Yvalues.get(j) == -1) {
						pw_n = pw_n + ReadFile.ProbabilityList.get(j);
					} else {
						pr_p = pr_p + ReadFile.ProbabilityList.get(j);
					}
				}
				for (int j = i + 1; j < ReadFile.n; j++) {
					if (ReadFile.Yvalues.get(j) == 1) {
						pw_p = pw_p + ReadFile.ProbabilityList.get(j);
					} else {
						pr_n = pr_n + ReadFile.ProbabilityList.get(j);
					}
				}

				//Calculate G Error
				double G = 0.0;
				G = Math.sqrt(pr_p * pw_n) + Math.sqrt(pw_p * pr_n);
				if (minG > G) {
					minG = G;
					x_pos = i;
					flag = true;
					final_prp=pr_p;
					final_prn = pr_n;
					final_pwp=pw_p;
					final_pwn = pw_n;
				}
			}

			for (int i = 0; i < ReadFile.n - 1; i++) {
				double pr_p = 0.0, pr_n = 0.0, pw_p = 0.0, pw_n = 0.0;
				for (int j = 0; j <= i; j++) {
					if (ReadFile.Yvalues.get(j) == -1) {
						pr_n = pr_n + ReadFile.ProbabilityList.get(j);
					} else {
						pw_p = pw_p + ReadFile.ProbabilityList.get(j);
					}
				}
				for (int j = i + 1; j < ReadFile.n; j++) {
					if (ReadFile.Yvalues.get(j) == 1) {
						pr_p = pr_p + ReadFile.ProbabilityList.get(j);
					} else {
						pw_n = pw_n + ReadFile.ProbabilityList.get(j);
					}
				}

				//Calculating G Error
				double G = 0.0;
				G = Math.sqrt(pr_p * pw_n) + Math.sqrt(pw_p * pr_n);
				if (minG > G) {
					minG = G;
					x_pos = i;
					flag = false;
					final_prp=pr_p;
					final_prn = pr_n;
					final_pwp=pw_p;
					final_pwn = pw_n;
				}
			}

			if (flag == true) {
				System.out.println("The selected weak classifier Ht: x < " + ((ReadFile.Xvalues.get(x_pos) + ReadFile.Xvalues.get(x_pos + 1)) / 2));
			} else {
				System.out.println("The selected weak classifier Ht: x > " + ((ReadFile.Xvalues.get(x_pos) + ReadFile.Xvalues.get(x_pos + 1)) / 2));
			}

			System.out.println("The G error value of Ht: " + minG);

			//Computing the classifier h(x)
			ArrayList<Integer> h = new ArrayList<>();
			if (flag == true) {
				for (int i = 0; i <= x_pos; i++) {
					h.add(1);
				}
				for (int i = x_pos + 1; i < ReadFile.n; i++) {
					h.add(-1);
				}
			}

			if (flag == false) {
				for (int i = 0; i <= x_pos; i++) {
					h.add(-1);
				}
				for (int i = x_pos + 1; i < ReadFile.n; i++) {
					h.add(1);
				}
			}

			//Calculating the weights Ct+ Ct- 
			double Ct_p = 0.0, Ct_n = 0.0;
			Ct_p = 0.5 * Math.log((final_prp + ReadFile.epsilon) / (final_pwn + ReadFile.epsilon));
			Ct_n = 0.5 * Math.log((final_pwp + ReadFile.epsilon) / (final_prn + ReadFile.epsilon));
			System.out.println("The weights Ct+, Ct-: " + Ct_p + ", " + Ct_n);

			//Set Pre normalized Factor P
			for (int i = 0; i < ReadFile.n; i++) {
				double tempP = 0.0;
				if (h.get(i) == 1) {
					tempP = ReadFile.ProbabilityList.get(i) * Math.pow(e, -(ReadFile.Yvalues.get(i) * Ct_p));
					ReadFile.ProbabilityList.set(i, tempP);
				} else {
					tempP = ReadFile.ProbabilityList.get(i) * Math.pow(e, -(ReadFile.Yvalues.get(i) * Ct_n));
					ReadFile.ProbabilityList.set(i, tempP);
				}
			}

			//Computing Normalization Factor Z
			for (int i = 0; i < ReadFile.n; i++) {
				z = z + ReadFile.ProbabilityList.get(i);
			}
			System.out.println("The probabilities normalization factor Zt: " + z);

			//Calculating New Probabilities
			for (int i = 0; i < ReadFile.n; i++) {
				double newp = ReadFile.ProbabilityList.get(i) / z;
				ReadFile.ProbabilityList.set(i, newp);
			}
			System.out.println("The probabilities after normalization: ");
			for (int i = 0; i < ReadFile.n; i++) {
				System.out.print(ReadFile.ProbabilityList.get(i) + " ");
			}
			System.out.println("");

			//Computing f(x)
			double tempf = 0.0;
			if (flag == true) {
				for (int i = 0; i <= x_pos; i++) {
					tempf = flist.get(i) + Ct_p;
					flist.set(i, tempf);
				}
				for (int i = x_pos + 1; i < ReadFile.n; i++) {
					tempf = flist.get(i) + Ct_n;
					flist.set(i, tempf);
				}
			}

			if (flag == false) {
				for (int i = 0; i <= x_pos; i++) {
					tempf = flist.get(i) + Ct_n;
					flist.set(i, tempf);
				}
				for (int i = x_pos + 1; i < ReadFile.n; i++) {
					tempf = flist.get(i) + Ct_p;
					flist.set(i, tempf);
				}
			}

			System.out.println("The values ft(xi) for each one of the examples: ");
			for (int i = 0; i < ReadFile.n; i++) {
				System.out.print(flist.get(i) + " ");

			}
			System.out.println("");

			//Calculating Boosted Error Et
			int err_count = 0;
			for (int i = 0; i < ReadFile.n; i++) {
				if (flist.get(i) > 0) {
					if (ReadFile.Yvalues.get(i) == -1) {
						err_count = err_count + 1;
					}
				} else {
					if (ReadFile.Yvalues.get(i) == 1) {
						err_count = err_count + 1;
					}
				}
			}

			double boostedErr = (double) err_count / (double) ReadFile.n;
			System.out.println("The Error of Boosted Classifier Et: " + boostedErr);

			//Computing Bound
			bound = bound * z;
			System.out.println("The Bound on Et: " + bound);
		}

	}
}




