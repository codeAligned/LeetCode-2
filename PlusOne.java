public class Solution {
    public int[] plusOne(int[] digits) {
		int len = digits.length;
		int tmp = 0;

		if (digits[len - 1] == 9) {
			tmp = 1;
			digits[len - 1] = 0;
		} else {
			digits[len - 1] = digits[len - 1] + 1;
		}

		for (int i = len - 2; i >= 0; i--) {
			if (tmp == 1) {
				if (digits[i] == 9) {
					digits[i] = 0;
				} else {
					digits[i] = digits[i] + 1;
					tmp = 0;
				}
			}
		}

		if (tmp == 1) {
			int[] res = new int[len + 1];
			res[0] = 1;
			for (int j = 1; j <= len; j++) {
				res[j] = digits[j-1];
			}
			return res;
		}else{
			return digits;
		}
		
	}
}