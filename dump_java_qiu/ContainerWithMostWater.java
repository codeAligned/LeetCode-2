public class Solution {
    public int maxArea(int[] height) {
        if (height == null || height.length <= 1) {
            return 0;
        }
        
        int start = 0;
        int end = height.length - 1;
        int maxArea = 0;
        while (start < end) {
            int curtHeight = Math.min(height[start], height[end]);
            maxArea = Math.max(maxArea, curtHeight * (end - start));
            if (height[start] < height[end]) {
                start++;
            } else {
                end--;
            }
        }
        return maxArea;
    }
}
