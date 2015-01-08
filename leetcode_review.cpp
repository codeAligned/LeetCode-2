#include <iostream>
#include <vector>
#include <stack>

using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int x) : val(x), left(NULL), right(NULL) {
    }
};

struct TreeLinkNode {
    int val;
    TreeLinkNode *left, *right, *next;

    TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {
    }
};

class Solution {
public:
    int singleNumber(int A[], int n) {
        int res = A[0];
        for (int i = 1; i < n; i++) {
            res = res ^ A[i];
        }
        return res;
    }

    int titleToNumber(string s) {
        int num = 0;
        for (string::iterator it = s.begin(); it != s.end(); ++it) {
            num = num * 26 + (*it - 'A' + 1);
        }
        return num;
    }

    vector<int> preOrderTranverse(TreeNode *root) {
        vector<int> result;

        //same as TreeNode const *p, const TreeNode* p
        const TreeNode *p;

        stack<const TreeNode *> s;

        p = root;
        if (p != nullptr) s.push(p);

        while (!s.empty()) {
            p = s.top();
            s.pop();
            result.push_back(p->val);

            if (p->right != nullptr) s.push(p->right);
            if (p->left != nullptr) s.push(p->right);
        }

        return result;
    }

    vector<int> inorderTraversal(TreeNode *root) {
        vector<int> result;
        const TreeNode *p;
        stack<const TreeNode *> s;

        p = root;

        while (!s.empty() || p != nullptr) {
            if (p != nullptr) {
                s.push(p);
                p = p->left;
            } else {
                p = s.top();
                s.pop();
                result.push_back(p->val);
                p = p->right;
            }
        }
        return result;

    }

    void connect(TreeLinkNode *root) {
        if (root == nullptr) return;

        TreeLinkNode *p, *head;
        p = root;
        head = p;

        while (head->left != nullptr) {
            while (p->next != nullptr) {
                p->left->next = p->right;
                p->right->next = p->next->left;
                p = p->next;
            }
            p->left->next = p->right;
            p = head->left;
            head = head->left;
        }
    }

    bool isSameTree(TreeNode *p, TreeNode *q) {
        if (p != nullptr && q != nullptr) {
            return (p->val == q->val) && isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
        } else {
            return p == nullptr && q == nullptr;
        }
    }

    vector<vector<int> > levelOrder(TreeNode *root) {
        vector<vector<int>> res;
        levelOrderDFS(0, root, res);
        return res;
    }

    void levelOrderDFS(int level, TreeNode *root, vector<vector<int>> &res) {
        if (!root) return;

        if (level >= res.size()) res.push_back(vector<int>());

        res[level].push_back(root->val);

        levelOrderDFS(level + 1, root->left, res);
        levelOrderDFS(level + 1, root->right, res);
    }

    TreeNode *sortedArrayToBST(vector<int> &num) {

    }

    int searchInsert(int A[], int n, int target) {
        if (n < 1 || target <= A[0]) return 0;

        for (int i = 1; i < n; ++i) {
            if (A[i] >= target) return i;
        }

        if (target > A[n - 1]) return n;
    }

};


int main() {
    Solution sol = Solution();
    int num = sol.titleToNumber("AA");
    cout << num << endl;
    return 0;
}