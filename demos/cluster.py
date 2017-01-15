MERGE_THRESHOLD = 0.2

class Cluster:
    """
    A cluster of face pic

    facepics: a set of facepic
    frame_id: a set of frame id that facepics belong to
    working: boolean status of cluster
    """

    def __init__(self, facepics):
        """
        :param facepics: a set of facepic
        """
        self.facepics = facepics
        self.frame_ids = set()
        self.working = True
        for facepic in facepics:
            self.frame_ids.add(facepic.frame_id)

    def merge(self, acluster):
        """
        merge 2 cluster, set 2 parent cluster working status to false
        :param acluster: another cluster
        :return:
        a cluster
        """
        facepics = self.facepics | acluster.facepics
        self.working = False
        acluster.working = False
        new_cluster = Cluster(facepics)
        return new_cluster

    def is_mergeable(self, acluster):
        """
        Check if this cluster is mergeable with acluster
        :param acluster: a cluster
        :return:
         True or False
        """
        collision = self.frame_ids & acluster.frame_ids
        ncollision = len(collision)
        a = float(ncollision) / len(self.frame_ids)
        b = float(ncollision) / len(acluster.frame_ids)
        c = max(a,b)
        if c < MERGE_THRESHOLD:
            return True
        else:
            return False
