

import java.io.{BufferedWriter, FileWriter}

import breeze.linalg.DenseVector
import breeze.numerics.{pow, sqrt}
import org.apache.spark.mllib.clustering.{BisectingKMeans, KMeans, KMeansModel}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.HashMap
import scala.util.control.Breaks


object Niharika_Gajam_Task2{

  def sum_vectors(vect1 : Vector,vect2 : Vector):(Vector)={
    var v1 = vect1.toDense
    var v2 = vect2.toDense
    val bv1 = new DenseVector(v1.toArray)
    val bv2 = new DenseVector(v2.toArray)
    val vectout = Vectors.dense((bv1 + bv2).toArray)
    vectout.toSparse
  }

  def euclidean_dist(vect1 : Vector,vect2 : Vector):(Double)={


    var v1 = vect1.toDense
    var v2 = vect2//.toDense  this is always centroid and in dense vector form always



    var i =0
    var sz = v1.size
    var sum = 0.0
    while(i < sz){
      val x = v1.apply(i) - v2.apply(i)
      sum = sum + pow(x,2)
      i = i+1
    }

    //println("sum is " + sum.toString)
    sqrt(sum)
  }

  def get_word(input_file : RDD[Seq[String]],hashingTF: HashingTF,idx : Int):(String)={
    var ans = "".toString


    val str_ans =  input_file.map(ip=>{
      ip.toList.foreach(s=>{
        if(hashingTF.indexOf(s.toString) == idx){
          ans = s
        }
      })
      (ans,1)
    }).groupByKey().map(x=>(x._1,1)).collect().toArray

    var str =  str_ans(0)._1


    var ind=0
    while(ind < str_ans.size){
      if(!str_ans(ind)._1.isEmpty){
        str = str_ans(ind)._1
      }
      ind = ind + 1
    }

    str
  }

  def main(args: Array[String]): Unit = {

    val spark_config = new SparkConf().setAppName("Kmeans Clustering").setMaster("local[1]")
    val spark_context = new SparkContext(spark_config)

    val input_string = args(0).toString//"/Users/nihagajam/Desktop/INF 553/HW4/inf553_assignment4/Data/yelp_reviews_clustering_small.txt".toString

    val input_file: RDD[Seq[String]] = spark_context.textFile(input_string)
      .map(_.split(" ").toSeq)

    var algotype = args(1).toString///"B".toString
    val N = args(2).toInt //8.toInt
    val numIterations = args(3).toInt //20

    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(input_file)

    tf.cache()

    val idf = new IDF().fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf).cache()






    if(algotype == "K"){
        var clusters = KMeans.train(tfidf, N, numIterations, 1, "kmeans()", 42)
       // var clusters = KMeans.train(tfidf, N, numIterations)

        // Evaluate clustering by computing Within Set Sum of Squared Errors
        val WSSSE = clusters.computeCost(tfidf)
       // println(s"Within Set Sum of Squared Errors = $WSSSE")

        val vectorsAndClusterIdx = tfidf.map{ point =>
          val prediction = clusters.predict(point)
          (point,prediction)
        }

        var cls =  vectorsAndClusterIdx.keyBy(_._2).groupByKey()


        var size_cluster = cls.map(x=>{
          (x._1,x._2.size)
        }).collect().toMap

        var cluster_elements_combine  = cls.map(x=>{
          var clus_num = x._1

          var it = x._2.toArray
          var sz = it.size

          var sum_vect = it(0)._1

          var i= 1
          while(i < sz){
            sum_vect = sum_vectors(sum_vect,it(i)._1)
            i= i+1
          }

          val total_sum_vect : Vector = Vectors.dense(sum_vect.toArray.map(x => x))

          (total_sum_vect.toSparse,clus_num)
        }).collect().toArray

        cluster_elements_combine.foreach(x=>{x._1})



        var top_freq_indices = HashMap.empty[Int, Set[Int]]
        var top_freq_words = HashMap.empty[Int, Set[String]]

        cluster_elements_combine.foreach(c=>{

          var v= c._1
          var vect = c._1.toArray

          var top_freq = vect.sortBy(x => -x).distinct.take(10)
          // println("freq in iteration : " + c._1.toString)

          top_freq.foreach(id=>{
           

            val loop = new Breaks;
            loop.breakable {

              v.foreachActive((index,value)=>{

                if(value == id){

                  if(!top_freq_indices.contains(c._2))
                    top_freq_indices(c._2) = Set(index)
                  else
                    top_freq_indices(c._2) = top_freq_indices(c._2) + index
                  loop.break;
                }

              })
            }

          })
        })


        top_freq_indices.foreach(x=>{

          var cluster_number = x._1
          var string_index_set= x._2

          string_index_set.foreach(idx =>{

            var str = get_word(input_file,hashingTF,idx)

            if(!top_freq_words.contains(cluster_number)){
              top_freq_words(cluster_number) = Set(str)
            }
            else{
              top_freq_words(cluster_number) = top_freq_words(cluster_number) + str
            }

          })

        })




        val esse_cal = tfidf.map{p=>
          val prediction = clusters.predict(p)
          val euclid_dist= euclidean_dist(clusters.clusterCenters(prediction),p)
          (prediction,euclid_dist*euclid_dist)
        }.reduceByKey{case(x,y)=>x+y}.collect().toMap


      


        val bw = new BufferedWriter(new FileWriter("Niharika_Gajam_Cluster_small_"+algotype+"_8_20.json"))
        bw.write("{ \n")
        bw.write("\t \"algorithm\": \"K-Means\", \n")
        bw.write("\t \"WSSE\": "+ WSSSE +", \n")
        bw.write("\t \"Clusters\": [ { \n")
        for (i <- 1 to N){
          if (i!= 1){
            bw.write("\t { \n")
          }
          bw.write("\t \t \"id\":  "+ i + ", \n")
          bw.write("\t \t \"size\": " + size_cluster(i-1) + ", \n")
          bw.write("\t \t \"error\": " + esse_cal(i-1) + ",\n")
          bw.write("\t \t \"terms\": [\"" + top_freq_words(i-1).toList.mkString("\",\"") + "\"]  \n")
          if (i == N) {
            bw.write("\t } ] \n")
          }
          else
            bw.write("\t }, \n")
        }
        bw.write("} \n")
        bw.close()
    }
    else if(algotype == "B"){

        val bkm = new BisectingKMeans().setK(N).setMaxIterations(numIterations).setSeed(42)
        var clusters = bkm.run(tfidf)

        //var clusters = KMeans.train(tfidf, N, numIterations)

        // Evaluate clustering by computing Within Set Sum of Squared Errors
        val WSSSE = clusters.computeCost(tfidf)
       // println(s"Within Set Sum of Squared Errors = $WSSSE")

        val vectorsAndClusterIdx = tfidf.map{ point =>
          val prediction = clusters.predict(point)
          (point,prediction)
        }

        var cls =  vectorsAndClusterIdx.keyBy(_._2).groupByKey()


        var size_cluster = cls.map(x=>{
          (x._1,x._2.size)
        }).collect().toMap

        var cluster_elements_combine  = cls.map(x=>{
          var clus_num = x._1

          var it = x._2.toArray
          var sz = it.size

          var sum_vect = it(0)._1

          var i= 1
          while(i < sz){
            sum_vect = sum_vectors(sum_vect,it(i)._1)
            i= i+1
          }

          val total_sum_vect : Vector = Vectors.dense(sum_vect.toArray.map(x => x))

          (total_sum_vect.toSparse,clus_num)
        }).collect().toArray

        cluster_elements_combine.foreach(x=>{x._1})



        var top_freq_indices = HashMap.empty[Int, Set[Int]]
        var top_freq_words = HashMap.empty[Int, Set[String]]

        cluster_elements_combine.foreach(c=>{

          var v= c._1
          var vect = c._1.toArray

          var top_freq = vect.sortBy(x => -x).distinct.take(10)
          // println("freq in iteration : " + c._1.toString)

          top_freq.foreach(id=>{
           

            val loop = new Breaks;
            loop.breakable {

              v.foreachActive((index,value)=>{

                if(value == id){

                  if(!top_freq_indices.contains(c._2))
                    top_freq_indices(c._2) = Set(index)
                  else
                    top_freq_indices(c._2) = top_freq_indices(c._2) + index
                  loop.break;
                }

              })
            }

          })
        })


        top_freq_indices.foreach(x=>{

          var cluster_number = x._1
          var string_index_set= x._2

          string_index_set.foreach(idx =>{

            var str = get_word(input_file,hashingTF,idx)

            if(!top_freq_words.contains(cluster_number)){
              top_freq_words(cluster_number) = Set(str)
            }
            else{
              top_freq_words(cluster_number) = top_freq_words(cluster_number) + str
            }

          })

        })




        val esse_cal = tfidf.map{p=>
          val prediction = clusters.predict(p)
          val euclid_dist= euclidean_dist(clusters.clusterCenters(prediction),p)
          (prediction,euclid_dist*euclid_dist)
        }.reduceByKey{case(x,y)=>x+y}.collect().toMap


      

        val bw = new BufferedWriter(new FileWriter("Niharika_Gajam_Cluster_small_"+algotype+"_8_20.json"))
        bw.write("{ \n")
        bw.write("\t \"algorithm\": \"Bisecting K-Means\", \n")
        bw.write("\t \"WSSE\": "+ WSSSE +", \n")
        bw.write("\t \"clusters\": [ { \n")
        for (i <- 1 to N){
          if (i!= 1){
            bw.write("\t { \n")
          }
          bw.write("\t \t \"id\":  "+ i + ", \n")
          bw.write("\t \t \"size\": " + size_cluster(i-1) + ", \n")
          bw.write("\t \t \"error\": " + esse_cal(i-1) + ",\n")
          bw.write("\t \t \"terms\": [\"" + top_freq_words(i-1).toList.mkString("\",\"") + "\"]  \n")
          if (i == N) {
            bw.write("\t } ] \n")
          }
          else
            bw.write("\t }, \n")
        }
        bw.write("} \n")
        bw.close()

    }







  }
}