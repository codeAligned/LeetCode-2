public class Solution {
    public static ArrayList<ArrayList<Integer>> permute(int[] num) {
        if(num.length <= 0)
            return null;
        return permute(num, num.length);
    }
       
    public static ArrayList<ArrayList<Integer>> permute(int[] num, int length){ 

        ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();

        if(length == 1){
            ArrayList<Integer> temp = new ArrayList<Integer>();
            temp.add(num[0]);
            res.add(temp);
            return res;
        }
        
        ArrayList<ArrayList<Integer>> prev = permute(num, length - 1);
        for(ArrayList<Integer> temp : prev){
            for(int i = 0; i <= temp.size(); i++){
                ArrayList<Integer> perm = new ArrayList<Integer>(temp);
                perm.add(i, num[length - 1]);
                res.add(perm);
            }
        }
        return res;
    }
}
