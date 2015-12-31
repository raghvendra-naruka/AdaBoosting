package ml;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.StringTokenizer;

//Reading File having Training Data for Real AdaBoosting
public class ReadFile {

	public static int T = 0;
	public static int n = 0;
	public static ArrayList<Double> Xvalues = new ArrayList<>();
	public static ArrayList<Integer> Yvalues = new ArrayList<>();
	public static ArrayList<Double> ProbabilityList = new ArrayList<>();
	public static Double epsilon;
	public ReadFile(String file1) throws FileNotFoundException
	{
		String path = System.getProperty("user.dir")
				+ (ReadFile.class.getPackage() == null ? "" : "\\"
						+ "\\src\\"
						+ ReadFile.class.getPackage().getName()
						.replace('.', '\\'));

		Scanner S = new Scanner(new File(path + "\\" + file1));

		String temp = S.nextLine();
		StringTokenizer token = new StringTokenizer(temp," ");

		T = Integer.parseInt(token.nextToken());
		n = Integer.parseInt(token.nextToken());
		epsilon = Double.parseDouble(token.nextToken());
		
		temp = "";
		temp = S.nextLine();
		StringTokenizer token1 = new StringTokenizer(temp," ");
		while(token1.hasMoreTokens()){
			Xvalues.add(Double.parseDouble(token1.nextToken()));
		}
		
		temp = "";
		temp = S.nextLine();
		StringTokenizer token2 = new StringTokenizer(temp," ");
		while(token2.hasMoreTokens()){
			Yvalues.add(Integer.parseInt(token2.nextToken()));
		}

		temp = "";
		temp = S.nextLine();
		StringTokenizer token3 = new StringTokenizer(temp," ");
		while(token3.hasMoreTokens()){
			ProbabilityList.add(Double.parseDouble(token3.nextToken()));
		}


	}
}
