package ml;
 
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

//Common or supporting methods 
public class CommonOperations {

	int totalItrs;
	int totalInstances;
	double epsilon;
	
	List<Instance> instanceList;
	List<Classification> hypothesis;
	
	@SuppressWarnings("resource")
	//Reading data from an Input File containing the Training Data 
	public ArrayList<Object> readInputFile(String inputFile) {
		System.out.println("inputFile: " + inputFile);
		Scanner scanner;
		ArrayList<Object> inputDataList = new ArrayList<>();
		try {
			scanner = new Scanner(new File(inputFile));
			
			totalItrs = scanner.nextInt();
			totalInstances = scanner.nextInt();
			epsilon = scanner.nextDouble();
			instanceList = new ArrayList<Instance>(totalInstances);
			 
			for(int i = 0; i < totalInstances; i++){
				Instance instance  = new Instance();
				instance.setAttr(scanner.nextDouble());
				instanceList.add(instance);
			}
			for(int i = 0; i < totalInstances; i++){
				instanceList.get(i).setTagetAttr(scanner.nextInt());
			}
			for(int i = 0; i < totalInstances; i++){
				instanceList.get(i).setPrb(scanner.nextDouble());
			}	

			inputDataList.add(instanceList);
			inputDataList.add(totalInstances);
			inputDataList.add(totalItrs);
			inputDataList.add(epsilon);
			
			
			System.out.println ("Total Number of Instances: " + totalInstances);
			System.out.println ("Total Number of Iterations: " + totalItrs);
			System.out.println ("epsilon: " + epsilon);
			
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}		

		
		return inputDataList;
	}
	
	//Gathering the Classification Data like Decision Points and Instance Lists
	public List<Classification> getClassificationData (List<Instance> instanceList) {
		hypothesis = new ArrayList<Classification>();
		double firstDecisionPoint = instanceList.get(0).getAttr()-1.0;
		hypothesis.add(new Classification (instanceList, firstDecisionPoint, true));
		hypothesis.add(new Classification (instanceList, firstDecisionPoint, false));
		
		for(int i = 0; i < instanceList.size()-1; i++){	
			double decisionPoint = (instanceList.get(i).getAttr() + instanceList.get(i+1).getAttr()) / 2.0 ; 
			hypothesis.add(new Classification (instanceList, decisionPoint, true));
			hypothesis.add(new Classification (instanceList, decisionPoint, false ));
		}
		
		double lastDecisionPoint = instanceList.get(instanceList.size()-1).getAttr()+1.0;
		hypothesis.add(new Classification (instanceList, lastDecisionPoint, true));
		hypothesis.add(new Classification (instanceList, lastDecisionPoint, false));
		return hypothesis;
	}
}
