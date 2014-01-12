public class Solution {
    public int romanToInt(String s) {
		HashMap<Character, Integer> roman = new HashMap<Character, Integer>();
		roman.put('I', 1);
		roman.put('V', 5);
		roman.put('X', 10);
		roman.put('L', 50);
		roman.put('C', 100);
		roman.put('D', 500);
		roman.put('M', 1000);
		int sum = 0;

		for (int i = 0; i < s.length(); i++) {
			if (i < s.length() - 1) {
				if (roman.get(s.charAt(i)) >= roman.get(s.charAt(i + 1))) {
					sum += roman.get(s.charAt(i));
				} else {
					i++;
					sum = sum + roman.get(s.charAt(i)) - roman.get(s.charAt(i-1));
				}
			} else {
				sum += roman.get(s.charAt(i));
			}
		}
		return sum;
	}
}