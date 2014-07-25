public class Solution {
    // volume[i] = [min(left[i], right[i]) - A[i]] * 1
    // volume[i] = min(max_left, max_right) - height
    public int trap(int[] A) {
        if (A == null || A.length < 2) {
            return 0;
        }
        int[] left = new int[A.length];
        int maxLeft = A[0];
        for (int i = 1; i < A.length; i++) {
            maxLeft = Math.max(maxLeft, A[i]);
            left[i] = maxLeft - A[i];
        }
        
        int sum = 0;
        int maxRight = A[A.length - 1];
        for (int i = A.length - 1; i >= 0; i--) {
            maxRight = Math.max(maxRight, A[i]);
            left[i] = Math.min(left[i], maxRight - A[i]);
            sum += left[i];
        }
        return sum;
    }
}
