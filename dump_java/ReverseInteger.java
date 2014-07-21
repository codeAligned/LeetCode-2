public class Solution {
    public int reverse(int x) {
		int symble;
		int res = 0;

		if (x >= 0) {
			symble = 1;
		} else {
			symble = -1;
		}

		int op_num = symble * x;
		if(op_num >= 10){
		int[] res_array = { symble, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0};
		int quotient = op_num;
		int tmp;
		int i = 1;
		while (quotient > 0) {
			tmp = quotient / 10;
			res_array[i] = quotient - tmp * 10;
			i++;
			quotient = tmp;
		}

		for (int j = 1; j < i; j++) {
			res += res_array[j] * Math.pow(10,(i-j-1));
		}

		return res*symble;
		} else {
		    return x;
		}
		
	}
}