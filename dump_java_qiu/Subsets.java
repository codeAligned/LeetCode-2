public class Solution {
    public List<List<Integer>> subsets(int[] S) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        result.add(new ArrayList<Integer>());
        if (S == null || S.length == 0) {
            return result;
        }
        // Sort the array first!!!
        Arrays.sort(S);
        List<Integer> solution = new ArrayList<Integer>();
        helpGetSubsets(S, 0, solution, result);
        return result;
    }
    
    private void helpGetSubsets(int[] S, int start, List<Integer> solution, List<List<Integer>> result) {
        for (int i = start; i < S.length; i++) {
            solution.add(S[i]);
            result.add(new ArrayList<Integer>(solution));
            helpGetSubsets(S, i + 1, solution, result);
            solution.remove(solution.size() - 1);
        }
    }
}
