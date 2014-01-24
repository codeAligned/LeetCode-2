public class Solution {
    public String intToRoman(int num) {

		ArrayList<String> res = new ArrayList<String>();
		int product = 0;

		while (num > 0) {
			if (num >= 1000) {
				product = num / 1000;
				for (int i = 0; i < product; i++) {
					res.add("M");
				}
				num = num - product * 1000;
			} else if (num >= 900 && num < 1000) {
				res.add("CM");
				num -= 900;
			} else if (num >= 500 && num < 900) {
				res.add("D");
				product = (num - 500) / 100;
				for (int i = 0; i < product; i++) {
					res.add("C");
				}
				num = num - 500 - product * 100;
			} else if (num >= 400 && num < 500) {
				res.add("CD");
				num -= 400;
			} else if (num >= 100 && num < 400) {
				product = num / 100;
				for (int i = 0; i < product; i++) {
					res.add("C");
				}
				num = num - product * 100;
			} else if (num >= 90 && num < 100) {
				res.add("XC");
				num -= 90;
			} else if (num >= 50 && num < 90) {
				res.add("L");
				product = (num - 50) / 10;
				for (int i = 0; i < product; i++) {
					res.add("X");
				}
				num = num - 50 - product * 10;
			} else if (num >= 40 && num < 50) {
				res.add("XL");
				num -= 40;
			} else if (num >= 10 && num < 40) {
				product = num / 10;
				for (int i = 0; i < product; i++) {
					res.add("X");
				}
				num = num - product * 10;
			} else if (num == 9) {
				res.add("IX");
				num = 0;
			} else if (num >= 5 && num < 9) {
				res.add("V");
				for (int i = 0; i < (num - 5); i++) {
					res.add("I");
				}
				num = 0;
			} else if (num == 4) {
				res.add("IV");
				num = 0;
			} else {
				for (int i = 0; i < num; i++) {
					res.add("I");
				}
				num = 0;
			}
		}

		String romanResult = "";
		
		for (String s : res){
			romanResult += s;
		}
		return romanResult;
	}
}