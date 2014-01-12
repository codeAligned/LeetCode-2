public class Solution {
    public int numTrees(int n) {
        int res = 0;
        if (n == 0)
            return 1;
        
        for (int i = 0; i < n; i++) {
            res += numTrees(n - 1 - i) * numTrees(i);
        }
        
        return res;
    }
}