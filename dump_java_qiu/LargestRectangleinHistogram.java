public class Solution {
    public int largestRectangleArea(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        
        // maintain an increasing stack
        Stack<Integer> indexStack = new Stack<Integer>();
        int i = 0;
        int maxArea = 0;
        
        while (i <= height.length) {
            int curtHeight = (i == height.length)? 0 : height[i];
            
            if (indexStack.isEmpty() || curtHeight > height[indexStack.peek()]) {
                indexStack.push(i); 
                i++;
            } else {
                int prevHeight = height[indexStack.pop()];
                int width = indexStack.isEmpty()? i : i - indexStack.peek() - 1;
                maxArea = Math.max(maxArea, prevHeight * width);
            }
        }
        return maxArea;
    }
}
