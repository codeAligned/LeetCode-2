public class Solution {
    public int singleNumber(int[] A) {
        int[] ret = new int[32];

		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < A.length; j++) {
				ret[i] += A[j] >> (31-i)&1;
			}
			ret[i] = ret[i] % 3;
		}

		int res = 0;
		for (int k = 0; k < 32; k++) {
			res += ret[31 - k] << k;
		}
		
		return res;
    }
}