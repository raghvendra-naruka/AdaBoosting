package ml;

import java.util.ArrayList;
import java.util.List;

//Classification Class
public class Classification {

	boolean isLowerPositive;
	double decisionPoint; 
	List<Instance> classifiedInstances;
	double epsilon,alpha;
	double G, cPlus, cMinus, rightPlus, rightMinus, wrongPlus, wrongMinus;
	
	public Classification(List<Instance> originalList, double decisionPoint, boolean isLowerPositive ) {
		this.isLowerPositive = isLowerPositive;
		this.decisionPoint = decisionPoint;
		this.classifiedInstances = classifyAllInstances(originalList);
	}
	
	//Computing the Goodness Weight of Hypothesis i.e. alpha 
	public double computeAlpha() {	
		alpha = 0.5 * Math.log( (1 - epsilon) / epsilon);
		return alpha;
	}
	
	//Calculating the Weighted Training Error of Hypothesis
	public void computeEpsilon(List<Instance> exampleSet){
	 epsilon = 0;
		for(int i = 0; i < exampleSet.size(); i++){
			if(classifiedInstances.get(i).getTagetAttr() != exampleSet.get(i).getTagetAttr()){
				epsilon += exampleSet.get(i).getPrb();
			}
		}
	}
	
	//Calculating the value of function G
	public double computeG(){
		G = Math.sqrt(rightPlus * wrongMinus) + Math.sqrt(wrongPlus * rightMinus);
		return G;
	}
	
	//Computing the value of weight C
	public void computeC(double  epsilon) {
		cPlus = 0.5 * Math.log((rightPlus + epsilon) / (wrongMinus + epsilon) );
		cMinus = 0.5 * Math.log((wrongPlus + epsilon) / (rightMinus + epsilon) );
	}
	
	//calculating Probabilities
	public void computeProbabilities(List<Instance>  exampleSet) {
		rightPlus = wrongPlus = rightMinus = wrongMinus = 0;
		for(int i = 0; i < exampleSet.size(); i++){	
			if(exampleSet.get(i).getTagetAttr() == 1){
				if(classifiedInstances.get(i).getTagetAttr()==1){
					rightPlus += exampleSet.get(i).getPrb();
				} else {
					wrongPlus += exampleSet.get(i).getPrb();
				}
			} else {
				if(classifiedInstances.get(i).getTagetAttr()==-1){
					rightMinus += exampleSet.get(i).getPrb();
				} else {
					wrongMinus += exampleSet.get(i).getPrb();
				}
			}
		}
	}
		
	private void classify(Instance instance) {
		instance.setTagetAttr(getTargetAttribute(instance));
	}
		
	//Classifying the Instances
	public List<Instance> classifyAllInstances(List<Instance> originalList) {
			classifiedInstances = new ArrayList<>(originalList.size());
			for(int i = 0; i < originalList.size(); i++){
				Instance instance = originalList.get(i).clone();
				classify(instance);
				classifiedInstances.add(instance);
			}
		return classifiedInstances;
	}

	private int getTargetAttribute(Instance instance) {
		if(instance.getAttr() < decisionPoint){
			return isLowerPositive ? 1 : -1;
		}	
	   return isLowerPositive ? -1 : 1;
	}
		
	//Cloning of Classifiers
	public Classification cloneClassifier() {
		Classification clonedClassifier  =  new Classification(classifiedInstances, decisionPoint, isLowerPositive);
		clonedClassifier.rightPlus = rightPlus; 
		clonedClassifier.wrongPlus = wrongPlus;
		clonedClassifier.rightMinus = rightMinus; 
		clonedClassifier.wrongMinus = wrongMinus;
	
		clonedClassifier.cPlus = cPlus;
		clonedClassifier.cMinus = cMinus;
		
		clonedClassifier.epsilon = epsilon;
		clonedClassifier.alpha = alpha;
		clonedClassifier.G = G;
		
		return clonedClassifier;
	}
	
	@Override
	public String toString() {
		if(isLowerPositive){
			return "x < " + decisionPoint;
		}
	  return "x > " + decisionPoint;
	}
	
}

