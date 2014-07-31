public class Solution {
    public String multiply(String num1, String num2) {
        if (num1 == null || num2 == null || num1.length() == 0
            || num2.length() == 0) {
            return "";
        }
        
        // new length is the sum of the length of two numbers
        int len = num1.length() + num2.length();
        int[] result = new int[len];
        int carry = 0;
        int index = 0;
        
        for (int i = num1.length() - 1; i >= 0; i--) {
            for (int j = num2.length() - 1; j >= 0; j--) {
                index = i + j + 1;
                result[index] += (num1.charAt(i) - '0') * (num2.charAt(j) - '0') + carry;
                carry = result[index] / 10;
                result[index] %= 10;
            }
            // Don't forget the last carry, after using set it to 0.
            if (carry > 0) {
                result[index - 1] = carry;
                carry = 0;
            }
        }
 
        String res = new String();
        for (int i = 0; i < result.length; i++) {
            res += String.valueOf(result[i]);
        }
        
        // remove the preceding '0's
        while (res.charAt(0) == '0' && res.length() > 1) {
            res = res.substring(1);
        }
        return res;
    }
}
