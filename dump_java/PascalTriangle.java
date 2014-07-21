public class Solution {
    public ArrayList<ArrayList<Integer>> generate(int numRows) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        
        if (numRows == 0) {
            return result; 
        }
        
        for (int j = 0; j< numRows; j++) {
        	ArrayList<Integer> thisRow = new ArrayList<Integer>(); 
        	for (int i = 0;i<=j;i++) {
        		if(i == 0)
        		    thisRow.add(1);
        		else if(i < j) {
        		    int element = result.get(j-1).get(i) + result.get(j-1).get(i-1);
        		    thisRow.add(element);
        		}else if (i == j) {
        		    thisRow.add(1);
        		}    
        	}
        	result.add(thisRow);
        }
        return result;
    }
}