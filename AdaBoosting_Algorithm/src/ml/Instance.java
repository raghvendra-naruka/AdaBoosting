package ml;

//Instance Class
public class Instance {
	private double attr;
	private int tagetAttr;
	private double prb;
	
	public double getAttr() {
		return attr;
	}

	public void setAttr(double attr) {
		this.attr = attr;
	}

	public int getTagetAttr() {
		return tagetAttr;
	}

	public void setTagetAttr(int tagetAttr) {
		this.tagetAttr = tagetAttr;
	}

	public double getPrb() {
		return prb;
	}

	public void setPrb(double prb) {
		this.prb = prb;
	}

	public Instance () {
		this.attr = 0;
		this.tagetAttr = 0;
		this.prb = 0;
	}

	public Instance (double attr, int tagetAttr, double prb) {
		this.attr = attr;
		this.tagetAttr = tagetAttr;
		this.prb = prb;
	}

	public Instance clone() {
		return new Instance (attr,tagetAttr,prb);
	}

	@Override
	public String toString() {
		return "x:" + attr + ", y:"
				+ tagetAttr + ", p:" + prb ;
	}
	
	
}
