package ml;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

//Main Class
public class MainDriver {
 
	@SuppressWarnings("rawtypes")
	public static void main(String[] args) throws FileNotFoundException {
		String INPUT_FILE = "";
		if(args.length < 1 ){
			System.out.println("Please enter path of input file: ");
			Scanner scanner = new Scanner(System.in);
			INPUT_FILE = scanner.next();
			scanner.close();
		} else {
			INPUT_FILE = args[0];
		}
		
		int totalItrs = 0;
		int totalInstances = 0;
		double epsilon = 0.0;
		
		ArrayList inputDataList = null;
		List<Instance> instanceList = null;
		List<Classification> hypothesis = null;
		CommonOperations opsObj = new CommonOperations() ;
		inputDataList = opsObj.readInputFile (INPUT_FILE);
		instanceList = extracted(inputDataList);
		totalInstances = (int) inputDataList.get(1);
		totalItrs = (int) inputDataList.get(2);
		epsilon = (double) inputDataList.get(3);
		hypothesis = opsObj.getClassificationData (instanceList);
		
		//Performing the Binary AdaBoosting algorithm
		System.out.println("\n");
		System.out.println("---------------------");
		System.out.println("Binary AdaBoosting");
		System.out.println("---------------------");
		BinaryAdaBoosting binAdaBoost = new BinaryAdaBoosting(instanceList, hypothesis, totalItrs, totalInstances, epsilon);
		binAdaBoost.performBinaryAdaboost(totalItrs);
		System.out.println("\n\n");
		
		//Performing the Real AdaBoosting algorithm
		System.out.println("---------------------");
		System.out.println("Real AdaBoosting");
		System.out.println("---------------------");
		RealAdaBoosting realAdaBoost = new RealAdaBoosting(instanceList, hypothesis, totalItrs, totalInstances, epsilon);
		realAdaBoost.performRealAdaBoost(totalItrs);
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	private static List<Instance> extracted(ArrayList inputDataList) {
		return (List<Instance>) inputDataList.get(0);
	}
	
}
