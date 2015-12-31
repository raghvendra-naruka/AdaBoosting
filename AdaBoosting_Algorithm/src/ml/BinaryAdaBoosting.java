package ml;

import java.util.LinkedList;
import java.util.List;

//Binary AdaBoosting
public class BinaryAdaBoosting {

	List<Instance> instanceSet;
	List<Classification> classifierSet;
	List<Classification> selectedClassifier;
	int totalInstances;
	double epsilon;
	
	public BinaryAdaBoosting(List<Instance> instanceList, List<Classification> hypothesis, int totalItrs, int totalInstances, double epsilon) {
		this.classifierSet = hypothesis;
		this.selectedClassifier = new LinkedList<>();
		this.totalInstances = totalInstances;
		this.epsilon = epsilon;
		this.instanceSet = instanceList;

	}
	
	//Selecting Classifier
	public Classification selectBestClassifier(){
		Classification bestClassifier = null;
		double minEpsilon = 1.0;
		for(int i = 0 ; i < classifierSet.size(); i++){
			Classification classifier = classifierSet.get(i);
			classifier.computeEpsilon(instanceSet);
			if(classifier.epsilon < minEpsilon ){
				minEpsilon = classifier.epsilon;
				bestClassifier = classifier;
			}
		}
		return bestClassifier;
	}

	//Updating Probabilities
	private double updateProbabilities(Classification classifier) {
		double normalizationFactor = 0, q;
		for(int i = 0; i < instanceSet.size(); i++){	
			if(instanceSet.get(i).getTagetAttr() == classifier.classifiedInstances.get(i).getTagetAttr()){
				q = Math.pow(Math.E, -1 * classifier.alpha);
			} else {
				q = Math.pow(Math.E, classifier.alpha);
			}
			instanceSet.get(i).setPrb (q*instanceSet.get(i).getPrb ());
			normalizationFactor += instanceSet.get(i).getPrb ();
		}	
		for(int i = 0; i < instanceSet.size(); i++){
			instanceSet.get(i).setPrb (instanceSet.get(i).getPrb () / normalizationFactor);
		}
		return normalizationFactor;
	}

	//Calculate Error Count
	private int getErrorsinBoostedClassifier() {
		int errorCount = 0;
		for(int i = 0; i < instanceSet.size(); i++){
			double targetAttribute = 0;
			for(Classification classifier : selectedClassifier){
				targetAttribute += classifier.alpha * classifier.classifiedInstances.get(i).getTagetAttr ();	
			}

			if((instanceSet.get(i).getTagetAttr ()==-1 && targetAttribute >= 0.0) || (instanceSet.get(i).getTagetAttr ()== 1 && targetAttribute < 0.0)	) {
				errorCount++;
			}
		}	
		return errorCount;
	}

	//Binary AdaBoosting calculations
	public void performBinaryAdaboost(int totalItrs){
		double boundOnNormalizationFactor = 1;

		for(int i = 1; i <= totalItrs; i++){
			Classification bestClassifier = selectBestClassifier();
			bestClassifier.computeAlpha();
			double normalizationFactor = updateProbabilities(bestClassifier);
			boundOnNormalizationFactor *= normalizationFactor;
			selectedClassifier.add(bestClassifier.cloneClassifier());	

			System.out.println("\nIteration " + i);
			System.out.println("The selected weak classifier: " + bestClassifier);
			System.out.println("The Error of Ht: " + bestClassifier.epsilon);
			System.out.println("The weight of Ht: " + bestClassifier.alpha);
			System.out.println("The probabilities normalization factor Zt: " + normalizationFactor);
			System.out.print("The probabilities after normalization: ");
			for(int j = 0; j < instanceSet.size(); j++){
				if (j==0) {
					System.out.print(instanceSet.get(j).getPrb ());
				} else {
					System.out.print(", "+ instanceSet.get(j).getPrb ());
				}

			}
			System.out.print("\nThe boosted classifier: ");
			int k=0;
			for(k = 0; k < selectedClassifier.size()-1; k++){
				System.out.print(selectedClassifier.get(k).alpha +"*" + selectedClassifier.get(k) + " + ");
			}
			
			System.out.print(selectedClassifier.get(k).alpha +"*" + selectedClassifier.get(k));
			System.out.println("\nThe error of the boosted classifier: " + (getErrorsinBoostedClassifier()/(double)instanceSet.size()));
			System.out.println("The bound on Et: " + boundOnNormalizationFactor);

		}
	}

}
